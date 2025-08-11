import os
import json
from tqdm import tqdm
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set your cuda device
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from torchvision import datasets

from lemminflect import getAllInflections

import argparse

def concepts2cnf(concepts, tokenizer, no_inflection=False):
    cnf = []
    # add left space and encode
    concept_set = set([tuple(tokenizer.encode(f' {x}')) for x in concepts])

    for concept in concepts:
        s = tuple(tokenizer.encode(f' {concept}'))
        inflections = set([s])
        if not no_inflection:
            for k, v in getAllInflections(concept).items():
                for x in v:
                    t = tuple(tokenizer.encode(f' {x}'))
                    if len(s) <= len(t) and t[:len(s)] == s:
                        continue
                    # when both surf and surfer are required concepts
                    # avoid the case that surfer is considered an inflection of surf
                    if t in concept_set:
                        continue
                    inflections.add(t)

        clause = tuple(inflections)
        cnf.append(clause)

    cnf = tuple(cnf)

    return cnf


parser = argparse.ArgumentParser(description='desc')
parser.add_argument('--qbit', type=int, help='linear quantization bit width. Conflict with `cluster`', default=0)
parser.add_argument('--cluster', type=int, help='normalized quantization, number of cluster. Conflict with `qbit`.', default=0)
parser.add_argument('--hmm_model_path', type=str, default='ctrlg/hmm_gpt2-large_common-gen_4096')
parser.add_argument('--output_path', type=str, default='result/output.json')
parser.add_argument('--dataset_path', type=str, default='data/common-gen_validation.json')
args = parser.parse_args()
qbit = args.qbit

BASE_MODEL_PATH = f'ctrlg/gpt2-large_common-gen'
# HMM_MODEL_PATH = f'ctrlg/hmm_gpt2-large_common-gen_4096'
# HMM_MODEL_PATH = f'ctrlg/hmm_gpt2-large_common-gen_32768'
HMM_MODEL_PATH = args.hmm_model_path
DATASET_PATH = args.dataset_path
OUTPUT_PATH = args.output_path

# load the pretrained base_model and HMM
print(">> loading models...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)
base_model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

hmm_model = ctrlg.HMM.from_pretrained(HMM_MODEL_PATH).to(device)
vocab_size = hmm_model.vocab_size
eos_token_id = hmm_model.eos_token_id
print(f"vocab size: {vocab_size}, eos token: {str(eos_token_id)}")

# hmm model PTQ
if args.cluster:
    cluster_ = args.cluster - 1 # 2^n - 1
    alpha_flow, beta_flow, gamma_flow = hmm_model.alpha_exp, hmm_model.beta.exp(), hmm_model.gamma.exp()
    scale_alpha = torch.max(alpha_flow) / cluster_
    alpha_flow.div_(scale_alpha).round_().clamp_(0, cluster_)
    scale_beta = torch.max(beta_flow) / cluster_
    beta_flow.div_(scale_beta).round_().clamp_(0, cluster_)
    scale_gamma = torch.max(gamma_flow) / cluster_
    gamma_flow.div_(scale_gamma).round_().clamp_(0, cluster_)
    pseudocount = 0.001
    alpha_flow += pseudocount / alpha_flow.shape[-1]
    beta_flow += pseudocount / beta_flow.shape[-1]
    gamma_flow += pseudocount / gamma_flow.shape[-1]
    alpha_flow.div_(torch.sum(alpha_flow, dim=-1, keepdim=True))
    beta_flow.div_(torch.sum(beta_flow, dim=-1, keepdim=True)).log_()
    gamma_flow.div_(torch.sum(gamma_flow, dim=0, keepdim=True)).log_()
    hmm_model.update_params(alpha_flow, beta_flow, gamma_flow)

# load weights from local. hmm_model.alpha_exp, hmm_model.beta, hmm_model.gamma
# hmm_model.alpha_exp.data = torch.load('data/alpha_exp8.pt', weights_only=True)
# hmm_model.beta.data = torch.load('data/beta8.pt', weights_only=True)
# hmm_model.gamma.data = torch.load('data/gamma8.pt', weights_only=True)
# hmm_model.to(torch.float32).to(device)

# constraints: prefix, suffix, wordcount, keyword, base model prompts
prefix = ''                 # generate text starting with nothing
suffix = '<|endoftext|>'    # generate text ending with '<|endoftext|>'; a suffix must end with the eos token
prompt = '<|endoftext|>'    # prompt the base model with the '<|endoftext|>' token
prefix_ids = tokenizer.encode(prefix)
suffix_ids = tokenizer.encode(suffix)
prompt_ids = tokenizer.encode(prompt)


with open(DATASET_PATH, 'r') as fin:
    dataset = json.load(fin)
    dataset_start = 0
    dataset_end = 992
    
# pre
tmp = {}
for example in dataset:
    idx = example['concept_set_idx']
    if dataset_start <= idx and idx <= dataset_end:
        tmp[idx] = {
            'concept_set_idx': idx,
            'concepts': example['concepts'],
            # 'target': example['target'],
            'sentences': [],
        }
dataset = [v for _, v in tmp.items()]
del tmp

print('generating process ...')
ac_builder = ctrlg.AhoCorasickBuilder(vocab_size)
word_count_builder = ctrlg.WordCountBuilder(tokenizer, vocab_size)
input_ids = torch.tensor([prompt_ids], device=device)

for idx in tqdm(range(0, len(dataset))):
    example = dataset[idx]
    concepts = example['concepts']
    cnf = concepts2cnf(concepts, tokenizer)
    # keyphrases = []
    # for words in cnf:
    #     keyphrases.append([tokenizer.decode(i) for i in words])
    # print(idx, keyphrases)
    dfa_graphs = []
    for keyphrase in cnf:
        dfa_graphs.append(ac_builder.build(keyphrase))
    # dfa_graphs.append(word_count_builder.build(10, 10))
    dfa_graph = ctrlg.DFA_prod(dfa_graphs, mode='intersection')
    dfa_model = ctrlg.DFAModel(dfa_graph, vocab_size).to(device)
    
    # logits processor: HMM + DFA
    min_new_tokens = 5
    max_new_tokens = 32
    constraint_logits_processor = ctrlg.ConstraintLogitsProcessor(
        hmm_model, dfa_model,
        min_new_tokens, max_new_tokens,  # the size without suffix and prefix
        prompt_ids, prefix_ids=prefix_ids, suffix_ids=suffix_ids, qbit=qbit)
    beam_size = 128  # beam search size. Generate several results
    constraint_logits_processor.hmm_batch_size = beam_size

    outputs = base_model.generate(
        input_ids=input_ids, do_sample=False, length_penalty=0.2,
        num_beams=beam_size, num_return_sequences=beam_size,
        min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList([constraint_logits_processor]),
        pad_token_id=tokenizer.eos_token_id,
    )

    # serialize the output
    generated_ids = ctrlg.extract_generated_ids( # removing prompt ids; remove suffix ids that are (partially) generated
        outputs.tolist(), prompt_ids, suffix_ids, eos_token_id)

    generated_ids = ctrlg.rank_generated_ids(base_model, generated_ids, prompt_ids, suffix_ids) # rank the generated ids by the base_model probability

    # dump outputs
    for generated in generated_ids:
        sentence = tokenizer.decode(prefix_ids, skip_special_tokens=True) + \
            tokenizer.decode(generated, skip_special_tokens=True) + \
            tokenizer.decode(suffix_ids, skip_special_tokens=True)
        dataset[idx]['sentences'].append(sentence.strip())

    if idx % 100 == 0:
        with open(OUTPUT_PATH, 'w') as fout:
            json.dump(dataset[:idx+1], fout, indent=2)

with open(OUTPUT_PATH, 'w') as fout:
    json.dump(dataset[:idx+1], fout, indent=2)