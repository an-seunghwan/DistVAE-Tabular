#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import namedtuple

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
#%%
class Model(nn.Module):
    def __init__(
        self, 
        EncodedInfo: namedtuple, # information of the dataset
        latent_dim: int = 8, # the latent dimension size
        beta: float = 0.1, # scale parameter of asymmetric Laplace distribution
        hidden_dim: int = 128, # hidden layer dimension
        
        epochs: int = 1000, # the number of epochs
        batch_size: int = 256, # batch size
        lr: float = 0.001, # learning rate
        
        threshold: float = 1e-8, # threshold for clipping alpha_tilde
        step: float = 0.1, # interval size of quantile levels (if step = 0.1, then M = 10)
        
        device="cpu"
    ):
        super(Model, self).__init__()
        
        self.EncodedInfo = EncodedInfo
        self.cont_dim = self.EncodedInfo.num_continuous_features
        self.disc_dim = sum(self.EncodedInfo.num_categories)
        self.p = self.cont_dim + self.disc_dim
        
        self.latent_dim = latent_dim
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.threshold = threshold
        self.step = step
        self.device = device
        
        """encoder"""
        self.encoder = nn.Sequential(
            nn.Linear(self.p, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2),
        ).to(device)
        
        """decoder"""
        self.delta = torch.arange(0, 1 + self.step, step=self.step).view(1, -1).to(device)
        self.M = self.delta.size(1) - 1
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.cont_dim * (1 + (self.M + 1)) + self.disc_dim),
        ).to(device)
    
    def get_posterior(self, input):
        h = self.encoder(input)
        mean, logvar = torch.split(h, self.latent_dim, dim=1)
        return mean, logvar
    
    def sampling(self, mean, logvar):
        noise = torch.randn(mean.size(0), self.latent_dim).to(self.device) 
        z = mean + torch.exp(logvar / 2) * noise
        return z
    
    def encode(self, input):
        mean, logvar = self.get_posterior(input)
        z = self.sampling(mean, logvar)
        return z, mean, logvar
    
    def quantile_parameter(self, z):
        h = self.decoder(z)
        logit = h[:, -self.disc_dim:] # categorical
        spline = h[:, :-self.disc_dim] # continuous
        h = torch.split(spline, 1 + (self.M + 1), dim=1)
        gamma = [h_[:, [0]] for h_ in h]
        beta = [torch.cat([
            torch.zeros_like(gamma[0]),
            nn.ReLU()(h_[:, 1:]) # positive constraint
        ], dim=1) for h_ in h]
        beta = [b[:, 1:] - b[:, :-1] for b in beta]
        return gamma, beta, logit
    
    def quantile_function(self, alpha, gamma, beta, j):
        return gamma[j] + (beta[j] * torch.where(
            alpha - self.delta > 0,
            alpha - self.delta,
            torch.zeros(()).to(self.device)
        )).sum(dim=1, keepdims=True)
        
    def _quantile_inverse(self, x, gamma, beta, j):
        delta_ = self.delta.unsqueeze(2).repeat(1, 1, self.M + 1)
        delta_ = torch.where(
            delta_ - self.delta > 0,
            delta_ - self.delta,
            torch.zeros(()).to(self.device))
        mask = gamma[j] + (beta[j] * delta_.unsqueeze(2)).sum(dim=-1).squeeze(0).t()
        mask = torch.where(
            mask <= x, 
            mask, 
            torch.zeros(()).to(self.device)).type(torch.bool).type(torch.float)
        alpha_tilde = x - gamma[j]
        alpha_tilde += (mask * beta[j] * self.delta).sum(dim=1, keepdims=True)
        alpha_tilde /= (mask * beta[j]).sum(dim=1, keepdims=True) + self.threshold
        alpha_tilde = torch.clip(alpha_tilde, self.threshold, 1) # numerical stability
        return alpha_tilde

    def quantile_inverse(self, x, gamma, beta):
        alpha_tilde_list = []
        for j in range(self.cont_dim):
            alpha_tilde = self._quantile_inverse(x[:, [j]], gamma, beta, j)
            alpha_tilde_list.append(alpha_tilde)
        return alpha_tilde_list
    
    def forward(self, input):
        z, mean, logvar = self.encode(input)
        gamma, beta, logit = self.quantile_parameter(z)
        return z, mean, logvar, gamma, beta, logit
    
    def gumbel_sampling(self, size, eps = 1e-20):
        U = torch.rand(size)
        G = (- (U + eps).log() + eps).log()
        return G
#%%