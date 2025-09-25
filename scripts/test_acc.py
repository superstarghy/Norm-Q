from eval_metrics.bleu import Bleu
from eval_metrics.cider import Cider
from eval_metrics.spice import Spice

import spacy
import json
import codecs
import argparse

from transformers import AutoTokenizer

from lemminflect import getAllInflections

nlp = spacy.load("en_core_web_sm")

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

def tokenize(dicts):
    for key in dicts:
        new_sentence_list = []
        for sentence in dicts[key]:
            a = ''
            for token in nlp(sentence):
                a += token.text
                a += ' '
            new_sentence_list.append(a.rstrip())
        dicts[key] = new_sentence_list

    return dicts


def evaluator(gts, res):
    eval = {}
    # =================================================
    # Set up scorers
    # =================================================
    print('tokenization...')
    # Todo: use Spacy for tokenization
    gts = tokenize(gts)
    res = tokenize(res)

    # =================================================
    # Set up scorers
    # =================================================
    print('setting up scorers...')
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"),
        # (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    # =================================================
    # Compute scores
    # =================================================
    for scorer, method in scorers:
        print("computing %s score..." % (scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            sum = 0
            for sc, scs, m in zip(score, scores, method):
                eval[m] = sc
                sum += sc
                print("%s: %0.3f" % (m, sc))
            print("average: %0.3f" % (sum / len(method)))
        else:
            eval[method] = score
            print("%s: %0.3f" % (method, score))
    
    return eval


def load_targets(dataset_file):
    with open(dataset_file, 'r') as fin:
        examples = json.load(fin)
        
    examples_ = {}
    for example in examples:
        idx = example['concept_set_idx']
        if idx in examples_:
            examples_[idx]['sentences'] = examples_[idx]['sentences'] + [example['target']]
        else:
            examples_[idx] = {
                'concept_set_idx': idx,
                'concepts': example['concepts'],
                'sentences': [example['target']],
            }
    
    examples = [v for _, v in examples_.items()]
    
    return examples


def check(sentence, concepts, tokenizer, log=0):
    cnf = concepts2cnf(concepts, tokenizer)
    for words in cnf: # all
        flag_any = False
        for w in words: # any 
            concept = tokenizer.decode(w).strip()
            if concept in sentence:
                flag_any = True
                break
        if flag_any == False:
            if log:
                print("concepts:", concepts)
                print("error sentence:", sentence)
                print('\n')
            return False
    return True


parser = argparse.ArgumentParser(description='desc')
parser.add_argument('--result', type=str, help='output file path', default='result/output.json')
parser.add_argument('--log', type=int, help='print the failed results', default=0)
args = parser.parse_args()

result_file = args.result
target_file = 'data/common-gen_validation.json'
targets = load_targets(target_file)
BASE_MODEL_PATH = f'ctrlg/gpt2-large_common-gen'
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

with open(result_file, 'r') as fin:
    results = json.load(fin)

targets = sorted(targets, key=lambda x:x['concept_set_idx'])
results = sorted(results, key=lambda x:x['concept_set_idx'])

results_idx_set = set([example['concept_set_idx'] for example in results])  # idx
targets = [example for example in targets if example['concept_set_idx'] in results_idx_set]

gts = {}
res = {}
hit = 0
print("size of dataset", len(targets), len(results))
for gts_line, res_line in zip(targets, results):
    assert(gts_line['concepts'] == res_line['concepts'])
    key = '#'.join(gts_line['concepts'])
    gts[key] = [x.rstrip('\n') for x in gts_line['sentences']]

    # sentence = res_line['sentence']
    sentence = res_line['sentences'][0]
    if check(sentence, res_line['concepts'], tokenizer, args.log):
        hit += 1
    # sentence.replace('.', ' .')
    # sentence.replace(',', ' ,')
    res[key] = [sentence.rstrip('\n')]

print("constraints accuracy: ", hit/len(results))

evaluator(gts, res)

from rouge_score import rouge_scorer
prediction = [x['sentences'][0] for x in results]
references = [x['sentences'] for x in targets]
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

scores = []
for pred, refs in zip(prediction, references):
    rs = [scorer.score(pred, i)['rougeL'].fmeasure for i in refs]
    scores.append(sum(rs)/len(rs))
print(f'average rougeL F score: {sum(scores) / len(scores)}')
