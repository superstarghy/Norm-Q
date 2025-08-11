#!/bin/bash
set -x

hmm_model_steps=("100" "50" "20" "10" "5" "2" "1" "0")

for hmm_model_step in "${hmm_model_steps[@]}"
do
python test_benchmark.py --hmm_model_path "distillation/workspace/models/\
hmm_gpt2-large_4096/checkpoint-q4s$hmm_model_step" --output_path "result/qat4_s$hmm_model_step.json"
done