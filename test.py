# import sys
# sys.path.append('../zeus')
# from zeus.monitor import ZeusMonitor

import os
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set your cuda device
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# export MKL_SERVICE_FORCE_INTEL=1
# export MKL_THREADING_LAYER=GNU
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MKL_THREADING_LAYER'] = 'GNU'

import torch
import ctrlg
print("import ctrlg")
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
# from torchvision import datasets

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
import argparse

print(">> import modules")

parser = argparse.ArgumentParser(description='desc')
parser.add_argument('--qbit', type=int, help='linear quantization bit width', default=0)
args = parser.parse_args()
qbit = args.qbit

# load the pretrained base_model and HMM
BASE_MODEL_PATH = f'ctrlg/gpt2-large_common-gen'
HMM_MODEL_PATH = f'ctrlg/hmm_gpt2-large_common-gen_4096'
# HMM_MODEL_PATH = './distillation/workspace/models/hmm_gpt2-large_4096/checkpoint-80'
# HMM_MODEL_PATH = f'ctrlg/hmm_gpt2-large_common-gen_32768'

print(">> loading models...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)
base_model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
print(">> loading hmm...")
hmm_model = ctrlg.HMM.from_pretrained(HMM_MODEL_PATH).to(torch.float32).to(device)
vocab_size = hmm_model.vocab_size
eos_token_id = hmm_model.eos_token_id
print(f"vocab size: {vocab_size}, eos token: {str(eos_token_id)}")

# hmm_model.alpha_exp, hmm_model.beta, hmm_model.gamma
# hmm_model.alpha_exp.data = torch.load('data/alpha_exp6.pt', weights_only=True)
# hmm_model.beta.data = torch.load('data/beta6.pt', weights_only=True)
# hmm_model.gamma.data = torch.load('data/gamma6.pt', weights_only=True)
# hmm_model.to(torch.float32).to(device)

# monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

# constraints: prefix, suffix, wordcount, keyword, base model prompts
prefix = ''                 # generate text starting with nothing
suffix = '<|endoftext|>'    # generate text ending with '<|endoftext|>'; a suffix must end with the eos token
prompt = '<|endoftext|>'    # prompt the base model with the '<|endoftext|>' token
prefix_ids = tokenizer.encode(prefix)
suffix_ids = tokenizer.encode(suffix)
prompt_ids = tokenizer.encode(prompt)

dfa_graphs = []
print(">> dfa...")


# constraint 1: keywords, AC automata
ac_builder = ctrlg.AhoCorasickBuilder(vocab_size)

# keyphrases = [
#     [' riding a bike', ' ride bikes', ' rides a bike', ' biking', ' bikes'],
#     [' park', ' beach', 'road']
# ]
keyphrases = [
    [' fields', ' fielded', ' fielding', ' field'],
    [' looks', ' looking', ' looked', ' look'],
    [' stands', ' stand', ' standing', ' stood']
]
# keyphrases = [
#     [' kid', ' kidded', ' kidding', ' kids'],
#     [' rooms', ' room'],
#     [' dancing', ' dances', ' danced', ' dance']
# ]
# keyphrases = [
#     [' shots', ' shot'],
#     [' gamed', ' gamer', ' games', ' gaming', ' gamest', ' game'],
#     [' players', ' player']
# ]
# keyphrases = [
#     [' pools', ' pooled', ' pool'],
#     [' suited', ' suiting', ' suit', ' suits'],
#     [' swim', ' swimming', ' swum', ' swam'],
# ]
for keyphrase in keyphrases:
    patterns = [tokenizer.encode(x) for x in keyphrase]
    dfa_graphs.append(ac_builder.build(patterns))

# constraint 2: word count
word_count_builder = ctrlg.WordCountBuilder(tokenizer, vocab_size)
a, b = 10, 10
dfa_graphs.append(word_count_builder.build(a, b))

# DFA operation & model
dfa_graph = ctrlg.DFA_prod(dfa_graphs, mode='intersection') # Intersection of the DFAs
dfa_model = ctrlg.DFAModel(dfa_graph, vocab_size).to(torch.float32).to(device) # for GPU execution


# logits processor: HMM + DFA
min_new_tokens = 5
max_new_tokens = 32
constraint_logits_processor = ctrlg.ConstraintLogitsProcessor(
    hmm_model, dfa_model,
    min_new_tokens, max_new_tokens,  # the size without suffix and prefix
    prompt_ids, prefix_ids=prefix_ids, suffix_ids=suffix_ids, qbit=qbit)

print(f"keyphrases: {keyphrases}")
print(f"word count: {a}~{b}")
beam_size = 128  # beam search size. Generate several results
constraint_logits_processor.hmm_batch_size = beam_size


# generate
input_ids = torch.tensor([prompt_ids], device=device)
print(">> generating...")
# monitor.begin_window("generating")
outputs = base_model.generate(
        input_ids=input_ids, do_sample=False, length_penalty=0.2,
        num_beams=beam_size, num_return_sequences=beam_size,
        min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList([constraint_logits_processor]),
        pad_token_id=tokenizer.eos_token_id,
    )
# mes = monitor.end_window("generating")
# print(f"Generaing consumed {mes.time} s and {mes.total_energy} J.")
'''
with profile(
    # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    activities=[ProfilerActivity.CPU],
    on_trace_ready=tensorboard_trace_handler('./log/ctrlg'),
    profile_memory=True,
    # record_shapes=True,
    # with_stack=True
) as prof:
    with record_function("base_model_generate"):
        prof.step()
        outputs = base_model.generate(
                input_ids=input_ids, do_sample=False, length_penalty=0.2,
                num_beams=beam_size, num_return_sequences=beam_size,
                min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens,
                logits_processor=LogitsProcessorList([constraint_logits_processor]),
                pad_token_id=tokenizer.eos_token_id,
            )

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
prof.export_chrome_trace("cputrace.json")
'''


# serialize the output
generated_ids = ctrlg.extract_generated_ids( # removing prompt ids; remove suffix ids that are (partially) generated
    outputs.tolist(), prompt_ids, suffix_ids, eos_token_id)

generated_ids = ctrlg.rank_generated_ids(base_model, generated_ids, prompt_ids, suffix_ids) # rank the generated ids by the base_model probability

# print top n outputs
for idx, generated in enumerate(generated_ids[:20]):
    print(f'{idx}. ' + tokenizer.decode(prefix_ids, skip_special_tokens=True) + \
          '\033[1m' + tokenizer.decode(generated, skip_special_tokens=True) + '\033[0m' + \
          tokenizer.decode(suffix_ids, skip_special_tokens=True))