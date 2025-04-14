import argparse
import os
import random
import scipy as sp
import pickle

import shutil
import csv
import ast
import gc


import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader


from NGG.autoencoders.autoencoder_base import VariationalAutoEncoder
from NGG.autoencoders.autoencoder_concat import VariationalAutoEncoder_concat
from NGG.autoencoders.autoencoder_GMVAE import GMVAE

from NGG.denoiser.denoise_model import DenoiseNN, p_losses, sample
from NGG.utils.utils import (
    linear_beta_schedule, 
    preprocess_dataset, 
)


from NGG.train_utils.parser import parse_train_arguments
from NGG.train_utils.load_or_not_deepset import load_or_not_deepset
from NGG.train_utils.load_or_not_stat_model import load_or_not_stat_model

from NGG.train_utils.load_autoencoder import load_autoencoder
from NGG.train_utils.train_autoencoder import train_autoencoder
from NGG.train_utils.denoiser_train import train_denoise
from NGG.train_utils.check_results import check_results

import sys

from torch.utils.data import Subset
np.random.seed(13)


args=parse_train_arguments()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#check which VAE model to use
VAE_mapper = {"base": VariationalAutoEncoder, "concat": VariationalAutoEncoder_concat, "features": 'NotImplemented', "GMVAE": GMVAE}
VAE_class = VAE_mapper[args.AE] 
if args.AE == 'GMVAE':
    args.labelize = True    
elif args.AE == 'concat':
    args.feature_concat= True

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.

print(f" dataset args : normalize : {args.normalize} \n labelize : {args.labelize} \n additional : {args.additional}")
if args.labelize:
    trainset, kmeans = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize,args.additional, n_clusters=args.n_clusters)
    validset, _ = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize,args.additional, kmeans=kmeans)
    testset, _ = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize,args.additional, kmeans=kmeans)
else:
    trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize,args.additional)
    validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize,args.additional)
    testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize,args.additional)

    kmeans = None

args.node_feature_dimension=trainset[0].x.shape[1]

# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

deepsets = load_or_not_deepset(args, device)
print(f"DeepSets state: {'Enabled' if deepsets else 'Disabled'}")
stat_model = load_or_not_stat_model(args, train_loader, device)
print(f"Stat model state: {'Enabled' if stat_model else 'Disabled'}")
to_labels = stat_model if stat_model is not None else kmeans
autoencoder = load_autoencoder(args, VAE_class,args.AE, to_labels, device, deepsets)

print(f"Autoencoder state: {'Enabled' if autoencoder else 'Disabled'}")
        
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)


autoencoder=train_autoencoder(args, autoencoder, train_loader, val_loader, device, optimizer, scheduler)

torch.cuda.empty_cache()
gc.collect()


# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# initialize denoising model
denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_condition, d_cond=args.dim_condition).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

denoise_model=train_denoise(args, denoise_model, autoencoder,optimizer,scheduler, train_loader, val_loader, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

denoise_model.eval()

del train_loader, val_loader

check_results(args, device, autoencoder, denoise_model, test_loader,testset,betas)