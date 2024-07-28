#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
#%%
from distvae_tabular.dataset import CustomDataset
from distvae_tabular.model import Model
#%%
def continuous_CRPS(model, x_batch, alpha_tilde_list, gamma, beta, j):
    term = (1 - model.delta.pow(3)) / 3 - model.delta - torch.maximum(alpha_tilde_list[j], model.delta).pow(2)
    term += 2 * torch.maximum(alpha_tilde_list[j], model.delta) * model.delta
    crps = (2 * alpha_tilde_list[j]) * x_batch[:, [j]]
    crps += (1 - 2 * alpha_tilde_list[j]) * gamma[j]
    crps += (beta[j] * term).sum(dim=1, keepdims=True)
    crps *= 0.5
    return crps.mean()
#%%
class DistVAE(nn.Module):
    def __init__(
        self, 
        data,
        continuous_features=[], 
        categorical_features=[], 
        integer_features=[], 
        ClfTarget=None,
        
        seed: int = 0, # seed for repeatable resultsseed for repeatable results
        latent_dim: int = 8, # the latent dimension size
        beta: float = 0.1, # scale parameter of asymmetric Laplace distribution
        hidden_dim: int = 128, # hidden layer dimension
        
        epochs: int = 1000, # the number of epochs
        batch_size: int = 256, # batch size
        lr: float = 0.001, # learning rate
        
        threshold: float = 1e-8, # threshold for clipping alpha_tilde
        step: float = 0.1, # interval size of quantile levels (if step = 0.1, then M = 10)
        lambda_: float = 0., # hyper-parameter for privacy control
        device="cpu"
    ):
        super(DistVAE, self).__init__()
        
        self.seed = seed
        self.latent_dim = latent_dim
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.threshold = threshold
        self.step = step
        self.lambda_ = lambda_
        self.device = device
        
        self.dataset = CustomDataset(
            data=data,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            integer_features=integer_features,
            ClfTarget=ClfTarget
        )
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size)
        self.EncodedInfo = self.dataset.EncodedInfo
        
        self.cont_dim = self.EncodedInfo.num_continuous_features
        self.disc_dim = sum(self.EncodedInfo.num_categories)
        self.p = self.cont_dim + self.disc_dim
        
        self.model = Model(
            EncodedInfo=self.EncodedInfo, # information of the dataset
            latent_dim=self.latent_dim, # the latent dimension size
            beta=self.beta, # scale parameter of asymmetric Laplace distribution
            hidden_dim=self.hidden_dim, # hidden layer dimension
            epochs=self.epochs, # the number of epochs
            batch_size=self.batch_size, # batch size
            lr=self.lr, # learning rate
            threshold=self.threshold, # threshold for clipping alpha_tilde
            step=self.step, # interval size of quantile levels (if step = 0.1, then M = 10)
            device=self.device
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr)
        
    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)       
        return 
    
    def train(self):
        self.set_random_seed(self.seed)
        
        for epoch in range(self.epochs):
            logs = {
                'loss': [], 
                'recon': [],
                'KL': [],
            }
            # for debugging
            logs['activated'] = []
            
            for x_batch in tqdm(iter(self.dataloader), desc="inner loop"):
                x_batch = x_batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                z, mean, logvar, gamma, beta, logit = self.model(x_batch)
                
                loss_ = []
                
                """1. Reconstruction loss: CRPS"""
                alpha_tilde_list = self.model.quantile_inverse(x_batch, gamma, beta)
                recon = 0
                for j in range(self.model.EncodedInfo.num_continuous_features):
                    recon += continuous_CRPS(self.model, x_batch, alpha_tilde_list, gamma, beta, j)
                st = 0
                cont_dim = self.model.EncodedInfo.num_continuous_features
                for j, dim in enumerate(self.model.EncodedInfo.num_categories):
                    ed = st + dim
                    _, targets = x_batch[:, cont_dim + st : cont_dim + ed].max(dim=1)
                    out = logit[:, st : ed]
                    recon += nn.CrossEntropyLoss()(out, targets)
                    st = ed
                loss_.append(('recon', recon))
                
                """2. KL-Divergence"""
                KL = torch.pow(mean, 2).sum(dim=1)
                KL -= logvar.sum(dim=1)
                KL += torch.exp(logvar).sum(dim=1)
                KL -= self.latent_dim
                KL *= 0.5
                KL = KL.mean()
                loss_.append(('KL', KL))
                
                """3. ELBO"""
                loss = recon + self.beta * KL 
                loss_.append(('loss', loss))
                
                var_ = torch.exp(logvar) < 0.1
                loss_.append(('activated', var_.float().mean()))
                
                loss.backward()
                self.optimizer.step()
                    
                """accumulate losses"""
                for x, y in loss_:
                    logs[x] = logs.get(x) + [y.item()]
            
            print_input = f"Epoch [{epoch+1:03d}/{self.epochs}]"
            print_input += "".join(
                [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
            )
            print(print_input)
            
        return
    
    def generate_data(self, n, seed=0):
        self.set_random_seed(seed)
        batch_size = 1024
        data = []
        steps = n // batch_size + 1
        
        for _ in tqdm(range(steps), desc="Generate Synthetic Dataset..."):
            with torch.no_grad():
                # prior distribution
                randn = torch.randn(
                    batch_size, self.latent_dim
                ).to(self.device) 
                gamma, beta, logit = self.model.quantile_parameter(randn)
                
                samples = []
                # continuous
                for j in range(self.EncodedInfo.num_continuous_features):
                    alpha = torch.rand(batch_size, 1).to(self.device)
                    
                    if self.lambda_ > 0:
                        ### The DistVAE Mechanism
                        u = torch.rand(batch_size, 1).to(self.device)
                        noise1 = self.lambda_ / (1. - alpha) * (u / alpha).log()
                        noise2 = self.lambda_ / (- alpha) * ((1. - u) / (1. - alpha)).log()
                        binary = (u <= alpha).to(float)
                        noise = noise1 * binary + noise2 * (1. - binary)
                        samples.append(self.model.quantile_function(alpha, gamma, beta, j) + noise) ### inverse transform sampling
                    elif self.lambda_ == 0:
                        samples.append(self.model.quantile_function(alpha, gamma, beta, j)) ### inverse transform sampling
                    else:
                        ValueError("lambda must be non-negative!")
                # categorical
                st = 0
                for j, dim in enumerate(self.EncodedInfo.num_categories):
                    ed = st + dim
                    out = logit[:, st : ed]
                    G = self.model.gumbel_sampling(out.shape).to(self.device)
                    _, out = (nn.LogSoftmax(dim=1)(out) + G).max(dim=1) ### Gumbel-Max Trick
                    samples.append(out.unsqueeze(1))
                    st = ed
                samples = torch.cat(samples, dim=1)
                data.append(samples)
                
        data = torch.cat(data, dim=0).to(float)
        data = data[:n, :]
        data = pd.DataFrame(
            data.cpu().numpy(), 
            columns=self.dataset.features)
        
        """un-standardization of synthetic data"""
        for col, scaler in self.dataset.scalers.items():
            data[[col]] = scaler.inverse_transform(data[[col]])
        
        """post-process"""
        data[self.dataset.categorical_features] = data[self.dataset.categorical_features].astype(int)
        data[self.dataset.integer_features] = data[self.dataset.integer_features].round(0).astype(int)
        
        return data
#%%