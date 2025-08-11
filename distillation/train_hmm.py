import os
import argparse
import random

import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from ctrlg import HMM
import faiss
import time
import random


def apply_dropout(input_ids, dropout, vocab_size, eos_token_id):
    n, d = input_ids.shape
    input_ids[torch.rand(n, device=input_ids.device) < dropout, -1] = eos_token_id
    input_ids[torch.logical_and(torch.rand(n, d, device=input_ids.device) < dropout, input_ids != eos_token_id)] = -1
    return input_ids

def Kmeans_faiss(vec:torch.Tensor, K=256, max_iterations=10, nredo=1, verbose=True, gpu_core=0):
    shape = vec.shape
    data = torch.flatten(vec).unsqueeze(1)
    kmeans = faiss.Kmeans(data.shape[1], K, niter=max_iterations, nredo=nredo, verbose=verbose,
        max_points_per_centroid=data.shape[0] // K, gpu=True)
    kmeans.train(data)
    centroids = kmeans.centroids
    D, I = kmeans.index.search(data, 1)
    transformed_data = centroids[I.flatten()]
    data = torch.tensor(transformed_data).reshape(shape)
    return data

def train_hmm(rank, world_size,
    model_path, checkpoint, save_per_step,
    data_path, dataset, total_chunks, batch_size, sample_length,
    em_schedule, log_file, dropout, pseudocount,
    quantization_step, quantization_bit_width, kmeans_flag):
    # --log_file ./workspace/logs/hmm_gpt2-large_4096_log.txt
    # --model_path ./workspace/models/hmm_gpt2-large_4096/ --checkpoint 0 
    # --data_path ./workspace/hmm_data/gpt2-large
    # --dataset gpt2-large
    # --save_per_step 10
    # --total_chunks 20
    # --batch_size 256
    # --em_schedule [(100, 1)] # [(10, 1), (5, 2), (4, 5), (4, 10), (4, 20)]
    # --dropout 0.01
    # --pseudocount 0.001
    device = f'cuda:{rank}'
    QTime = 0

    hmm_model = HMM.from_pretrained(f'{model_path}/checkpoint-{checkpoint}', map_location='cpu').to(device)
    hidden_states, vocab_size, eos_token_id = hmm_model.hidden_states, hmm_model.vocab_size, hmm_model.eos_token_id
    eps_cuda = torch.tensor([1e-7], device=device)

    dev_data = torch.load(f'{args.data_path}/{dataset}.dev')[:, :sample_length]
    dev_size = dev_data.shape[0]
    num_per_process = dev_data.shape[0] // world_size + 1
    dev_data = dev_data[rank * num_per_process: min(dev_data.shape[0], (rank+1) * num_per_process)]

    random_permutation = random.sample(range(0, total_chunks), total_chunks)
    for _, step_size in em_schedule:
        assert step_size <= total_chunks

    step_offset = checkpoint
    for step_count, step_size in em_schedule:
        for step_idx in range(0, step_count):
            # evaluate ll
            if step_offset == checkpoint:
                dev_ll = hmm_model.loglikelihood(dev_data, batch_size)
                torch.distributed.all_reduce(dev_ll, op=dist.ReduceOp.SUM)
                if rank == 0:
                    dev_ll = dev_ll.item() / dev_size
                    msg = f'{checkpoint}\t{-1.0}\t{dev_ll}'
                    print(msg)
                    with open(log_file, 'a+') as fout:
                        fout.write(msg + '\n')

            # get train_step for current step
            train_step = torch.cat([torch.load(f'{data_path}/{dataset}.train.{random_permutation[idx % total_chunks]}')
                for idx in range(step_offset, step_offset+step_size)], dim=0)
            train_step = train_step[torch.randperm(train_step.size(0))] # shuffle

            # get train_data for current process
            num_per_process = train_step.shape[0] // world_size + 1
            train_data = train_step[rank * num_per_process: min(train_step.shape[0], (rank+1) * num_per_process)]
            train_data_eval = train_data[:dev_data.shape[0]].clone()
            train_data = apply_dropout(train_data, dropout, vocab_size, eos_token_id)

            # compute flows for one em step
            alpha_flow = torch.zeros(hidden_states, hidden_states, device=device)
            beta_flow = torch.zeros(vocab_size + 1, hidden_states, device=device) # one extra token_id to account for MISSING token
            gamma_flow = torch.zeros(hidden_states, device=device)

            with torch.no_grad():
                for batch_idx in tqdm(range(0, train_data.shape[0], batch_size)):
                    batch_size_ = min(batch_size, train_data.shape[0] - batch_idx)
                    train_data_batch = train_data[batch_idx: batch_idx + batch_size_].to(device)

                    probs = hmm_model.forward(train_data_batch)
                    hmm_model.backward(train_data_batch, probs,
                        alpha_flow, beta_flow, gamma_flow)

                alpha_flow.mul_(hmm_model.alpha_exp)

                # distribute flow on the MISSING token
                missing_flow = beta_flow[vocab_size, :] / (vocab_size-1)
                beta_flow.add_(missing_flow[None, :])
                beta_flow[eos_token_id, :] -= missing_flow
                beta_flow[eos_token_id, :] = torch.maximum(beta_flow[eos_token_id, :], eps_cuda)

                beta_flow = torch.permute(beta_flow[:-1, :], (1, 0)).contiguous() # hidden_states * vocab_size

            torch.distributed.all_reduce(alpha_flow, op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(beta_flow, op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(gamma_flow, op=dist.ReduceOp.SUM)

            with torch.no_grad():
                q_start_time = time.time()
                if quantization_step and ((step_offset + step_size) % quantization_step == 0): # quantization
                    alpha_flow.div_(torch.sum(alpha_flow, dim=-1, keepdim=True))
                    beta_flow.div_(torch.sum(beta_flow, dim=-1, keepdim=True))
                    gamma_flow.div_(torch.sum(gamma_flow, dim=0, keepdim=True))
                    # quantization
                    if kmeans_flag:
                        # (1) Kmeans
                        alpha_flow = Kmeans_faiss(alpha_flow.cpu(), 2 ** quantization_bit_width, 10).to(device)
                        beta_flow = Kmeans_faiss(beta_flow.cpu(), 2 ** quantization_bit_width, 10).to(device)
                        gamma_flow = Kmeans_faiss(gamma_flow.cpu(), 2 ** quantization_bit_width, 10).to(device)
                    else:
                        # (2) Linear
                        cluster = 2 ** quantization_bit_width - 1
                        scale = torch.max(alpha_flow) / cluster
                        alpha_flow.div_(scale).round_().clamp_(0, cluster)
                        scale = torch.max(beta_flow) / cluster
                        beta_flow.div_(scale).round_().clamp_(0, cluster)
                        scale = torch.max(gamma_flow) / cluster
                        gamma_flow.div_(scale).round_().clamp_(0, cluster)
                q_end_time = time.time()
                QTime += q_end_time - q_start_time
                # flow to params; in-place to reduce memory consumption
                alpha_flow += pseudocount / alpha_flow.shape[-1]
                beta_flow += pseudocount / beta_flow.shape[-1]
                gamma_flow += pseudocount / gamma_flow.shape[-1]
                alpha_flow.div_(torch.sum(alpha_flow, dim=-1, keepdim=True))
                beta_flow.div_(torch.sum(beta_flow, dim=-1, keepdim=True)).log_()
                gamma_flow.div_(torch.sum(gamma_flow, dim=0, keepdim=True)).log_()
                hmm_model.update_params(alpha_flow, beta_flow, gamma_flow)

                # evaluate ll
                train_ll = hmm_model.loglikelihood(train_data_eval, batch_size)
                dev_ll = hmm_model.loglikelihood(dev_data, batch_size)

            torch.distributed.all_reduce(train_ll, op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(dev_ll, op=dist.ReduceOp.SUM)

            if rank == 0:
                train_ll = train_ll.item() / dev_size
                dev_ll = dev_ll.item() / dev_size
                ckpt = step_offset + step_size
                msg = f'{ckpt}\t{train_ll}\t{dev_ll}'
                print(msg)
                with open(log_file, 'a+') as fout:
                    fout.write(msg + '\n')

                if ckpt % save_per_step == 0 and ckpt != 0:
                    hmm_model.save_pretrained(f'{model_path}/checkpoint-{ckpt}')

            step_offset += step_size

            torch.cuda.empty_cache()
    print(f"Time overhead: {QTime: .2f}s")


if __name__ == '__main__':
    torch.cuda.empty_cache()

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--model_path', default='', type=str)
    arg_parser.add_argument('--checkpoint', default=50, type=int)
    arg_parser.add_argument('--save_per_step', default=150, type=int)

    arg_parser.add_argument('--data_path', default='', type=str)
    arg_parser.add_argument('--dataset', default='', type=str)
    arg_parser.add_argument('--total_chunks', default=200, type=int)
    arg_parser.add_argument('--batch_size', default=32, type=int)
    arg_parser.add_argument('--sample_length', default=None, type=int)
    arg_parser.add_argument('--em_schedule', type=str)
    arg_parser.add_argument('--dropout', default=0.001, type=float)
    arg_parser.add_argument('--pseudocount', default=0.001, type=float)

    arg_parser.add_argument('--log_file', default='', type=str)
    arg_parser.add_argument('--quantization_step', default=0, type=int)
    arg_parser.add_argument('--quantization_bit_width', default=8, type=int)
    arg_parser.add_argument('--kmeans', action='store_true')

    args = arg_parser.parse_args()

    dist.init_process_group('nccl')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        with open(args.log_file, 'a+') as fout:
            fout.write(str(vars(args)) + '\n')

    em_schedule = [tuple([int(y) for y in x.split(',')]) for x in args.em_schedule.split(';') if x != '']

    start_time = time.time()
    start_memory = torch.cuda.memory_allocated()

    train_hmm(rank, world_size,
        args.model_path, args.checkpoint, args.save_per_step,
        args.data_path, args.dataset, args.total_chunks, args.batch_size, args.sample_length,
        em_schedule, args.log_file, args.dropout, args.pseudocount,
        args.quantization_step, args.quantization_bit_width, args.kmeans)
    
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated()
    elapsed_time = end_time - start_time
    gpu_memory_usage = end_memory - start_memory
    print(f"training time: {elapsed_time:.2f} s")
    print(f"GPU Memory Usage: {gpu_memory_usage / (1024 ** 2):.2f} MB")