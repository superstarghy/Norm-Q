import os
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set your cuda device
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from torchvision import datasets

from ctrlg.mytest import *

from sklearn.cluster import KMeans

HMM_MODEL_PATH = f'ctrlg/hmm_gpt2-large_common-gen_4096'
# HMM_MODEL_PATH = f'ctrlg/hmm_gpt2-large_common-gen_32768'

hmm_model = ctrlg.HMM.from_pretrained(HMM_MODEL_PATH)
vocab_size = hmm_model.vocab_size
eos_token_id = hmm_model.eos_token_id
alpha_exp, beta, gamma = hmm_model.alpha_exp, hmm_model.beta, hmm_model.gamma
print(alpha_exp.shape, beta.shape, gamma.shape)
# print("alpha_exp", hmm_model.alpha_exp.shape, hmm_model.alpha_exp)
# print("beta", hmm_model.beta.shape, hmm_model.beta)
# print("gamma", hmm_model.gamma.shape, hmm_model.gamma)

b = 6
r = 2 ** b

shape = gamma.shape
kmeans = KMeans(n_clusters=r, random_state=0).fit(gamma.reshape([-1, 1]))
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
gamma = torch.tensor(centroids[labels]).reshape(shape)
print("gamma", gamma.shape, gamma)
torch.save(gamma, f'data/gamma{b}.pt')
torch.save(torch.tensor(labels), f'data/gamma{b}labels.pt')
torch.save(torch.tensor(centroids), f'data/gamma{b}centroids.pt')
loaded_tensor = torch.load(f'data/gamma{b}.pt', weights_only=True)
assert((loaded_tensor == gamma).all())
print("Loaded Tensor:", loaded_tensor)

shape = alpha_exp.shape
kmeans = KMeans(n_clusters=r, random_state=0).fit(alpha_exp.reshape([-1, 1]))
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
alpha_exp = torch.tensor(centroids[labels]).reshape(shape)
print("alpha_exp", alpha_exp.shape, alpha_exp)
torch.save(alpha_exp, f'data/alpha_exp{b}.pt')
torch.save(torch.tensor(labels), f'data/alpha_exp{b}labels.pt')
torch.save(torch.tensor(centroids), f'data/alpha_exp{b}centroids.pt')
loaded_tensor = torch.load(f'data/alpha_exp{b}.pt', weights_only=True)
assert((loaded_tensor == alpha_exp).all())
print("Loaded Tensor:", loaded_tensor)

shape = beta.shape
kmeans = KMeans(n_clusters=r, random_state=0).fit(beta.reshape([-1,1]))
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
# print("labels", labels)
# print("centroids", centroids)
beta = torch.tensor(centroids[labels]).reshape(shape)
print("beta", beta.shape, beta)
torch.save(beta, f'data/beta{b}.pt')
torch.save(torch.tensor(labels), f'data/beta{b}labels.pt')
torch.save(torch.tensor(centroids), f'data/beta{b}centroids.pt')
loaded_tensor = torch.load(f'data/beta{b}.pt', weights_only=True)
assert((loaded_tensor == beta).all())
print("Loaded Tensor:", loaded_tensor)