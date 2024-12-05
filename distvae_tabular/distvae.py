# %%
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from distvae_tabular.dataset import CustomDataset
from distvae_tabular.model import Model


# %%
def CRPS_loss(model, x_batch, alpha_tilde, gamma, beta):
    C = model.EncodedInfo.num_continuous_features
    delta = model.delta.unsqueeze(-1)  # [1, M+1, 1]

    term = (
        (1 - delta.pow(3)) / 3
        - delta
        - torch.maximum(alpha_tilde.unsqueeze(1), delta).pow(2)
    )  # [batch, M+1, p]
    term += (
        2 * torch.maximum(alpha_tilde.unsqueeze(1), delta) * delta
    )  # [batch, M+1, p]

    crps = (2 * alpha_tilde) * x_batch[:, :C]  # [batch, p]
    crps += (1 - 2 * alpha_tilde) * torch.cat(gamma, dim=1)  # [batch, p]
    crps += (torch.stack(beta, dim=2) * term).sum(dim=1)  # [batch, p]
    crps *= 0.5
    return crps.mean(dim=0).sum()


# %%
class DistVAE(nn.Module):
    def __init__(
        self,
        data: pd.DataFrame,
        continuous_features=[],
        categorical_features=[],
        integer_features=[],
        seed: int = 0,
        latent_dim: int = 8,
        beta: float = 0.1,
        hidden_dim: int = 128,
        epochs: int = 500,
        batch_size: int = 256,
        lr: float = 0.001,
        step: float = 0.1,
        threshold: float = 1e-8,
        device="cpu",
    ):
        """
        Args:
            data (pd.DataFrame): the observed tabular dataset
            continuous_features (list, optional): the list of continuous columns of data. Defaults to [].
                - If it is [], then all columns of data will be treated as continuous column
            categorical_features (list, optional): the list of categorical columns of data. Defaults to [].
                - If it is [], all other columns except continuous columns will be categorical column.
            integer_features (list, optional): the list of integer-type columns of data. Defaults to [].

            seed (int, optional): seed for repeatable results. Defaults to 0.
            latent_dim (int, optional): the latent dimension size. Defaults to 8.
            beta (float, optional): scale parameter of asymmetric Laplace distribution. Defaults to 0.1.
            hidden_dim (int, optional): the number of nodes in MLP. Defaults to 128.

            epochs (int, optional): the number of epochs. Defaults to 1000.
            batch_size (int, optional): the batch size. Defaults to 256.
            lr (float, optional): learning rate. Defaults to 0.001.

            step (float, optional): interval size between knots. Defaults to 0.1.
            threshold (float, optional): threshold for clipping alpha_tild (numerical stability). Defaults to 1e-8.
            device (str, optional): device. Defaults to "cpu".
        """

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
        self.device = device

        self.dataset = CustomDataset(
            data=data,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            integer_features=integer_features,
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        self.EncodedInfo = self.dataset.EncodedInfo
        self.cont_dim = self.EncodedInfo.num_continuous_features
        self.disc_dim = sum(self.EncodedInfo.num_categories)
        self.p = self.cont_dim + self.disc_dim

        self.set_random_seed(self.seed)
        self.initialize()

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        return

    def initialize(self):
        self.model = Model(
            EncodedInfo=self.EncodedInfo,  # information of the dataset
            latent_dim=self.latent_dim,  # the latent dimension size
            beta=self.beta,  # scale parameter of asymmetric Laplace distribution
            hidden_dim=self.hidden_dim,  # hidden layer dimension
            epochs=self.epochs,  # the number of epochs
            batch_size=self.batch_size,  # batch size
            lr=self.lr,  # learning rate
            threshold=self.threshold,  # threshold for clipping alpha_tilde
            step=self.step,  # interval size of quantile levels (if step = 0.1, then M = 10)
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return

    def train(self):
        self.set_random_seed(self.seed)

        for epoch in tqdm(range(self.epochs), desc="Training..."):
            logs = {
                "loss": [],
                "recon": [],
                "KL": [],
                "activated": [],
            }

            for x_batch in iter(self.dataloader):
                x_batch = x_batch.to(self.device)

                self.optimizer.zero_grad()

                z, mean, logvar, gamma, beta, logit = self.model(x_batch)

                loss_ = []

                """1. Reconstruction loss"""
                ### continuous: CRPS
                alpha_tilde = self.model.quantile_inverse(x_batch, gamma, beta)
                recon = CRPS_loss(self.model, x_batch, alpha_tilde, gamma, beta)
                ### categorical: classification loss
                st = 0
                cont_dim = self.model.EncodedInfo.num_continuous_features
                for dim in self.model.EncodedInfo.num_categories:
                    ed = st + dim
                    _, targets = x_batch[:, cont_dim + st : cont_dim + ed].max(dim=1)
                    out = logit[:, st:ed]
                    recon += nn.CrossEntropyLoss()(out, targets)
                    st = ed
                loss_.append(("recon", recon))

                """2. KL-Divergence"""
                KL = torch.pow(mean, 2).sum(dim=1)
                KL -= logvar.sum(dim=1)
                KL += torch.exp(logvar).sum(dim=1)
                KL -= self.latent_dim
                KL *= 0.5
                KL = KL.mean()
                loss_.append(("KL", KL))

                """3. ELBO"""
                loss = recon + self.beta * KL
                loss_.append(("loss", loss))

                ### check the size of active latent subspace
                # An, S., & Jeon, J. J. (2024).
                # Customization of latent space in semi-supervised Variational AutoEncoder.
                # Pattern Recognition Letters, 177, 54-60.
                var_ = torch.exp(logvar) < 0.1
                loss_.append(("activated", var_.float().mean()))

                loss.backward()
                self.optimizer.step()

                for x, y in loss_:
                    logs[x] = logs.get(x) + [y.item()]

            print_input = f"Epoch [{epoch+1:03d}/{self.epochs}]"
            print_input += "".join(
                [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
            )
            print(print_input)

        return

    def generate_data(
        self,
        n: int,
        lambda_: float = 0.0,
        seed: int = 0,
    ):
        """
        Args:
            n (int): the number of synthetic samples to generate
            lambda_ (float, optional): the hyper-parameter for privacy control (it must be non-negative). Defaults to 0.
            seed (int, optional): seed for repeatable results. Defaults to 0.
        """

        if lambda_ < 0:
            ValueError("lambda must be non-negative!")

        self.set_random_seed(seed)
        batch_size = 1024
        data = []
        steps = n // batch_size + 1

        for _ in tqdm(range(steps), desc="Generate Synthetic Dataset..."):
            with torch.no_grad():
                # prior distribution
                randn = torch.randn(batch_size, self.latent_dim).to(self.device)
                gamma, beta, logit = self.model.quantile_parameter(randn)

                samples = []
                # continuous
                for j in range(self.EncodedInfo.num_continuous_features):
                    alpha = torch.rand(batch_size, 1).to(self.device)

                    if lambda_ > 0:
                        ### The DistVAE Mechanism
                        u = torch.rand(batch_size, 1).to(self.device)
                        noise1 = lambda_ / (1.0 - alpha) * (u / alpha).log()
                        noise2 = lambda_ / (-alpha) * ((1.0 - u) / (1.0 - alpha)).log()
                        binary = (u <= alpha).to(float)
                        noise = noise1 * binary + noise2 * (1.0 - binary)
                    else:
                        noise = 0.0
                    samples.append(
                        self.model.quantile_function(alpha, gamma, beta, j) + noise
                    )  ### inverse transform sampling
                # categorical
                st = 0
                for j, dim in enumerate(self.EncodedInfo.num_categories):
                    ed = st + dim
                    out = logit[:, st:ed]
                    G = self.model.gumbel_sampling(out.shape).to(self.device)
                    _, out = (nn.LogSoftmax(dim=1)(out) + G).max(
                        dim=1
                    )  ### Gumbel-Max Trick
                    samples.append(out.unsqueeze(1))
                    st = ed
                samples = torch.cat(samples, dim=1)
                data.append(samples)

        data = torch.cat(data, dim=0).to(float)
        data = data[:n, :]
        data = pd.DataFrame(data.cpu().numpy(), columns=self.dataset.features)

        # un-standardization of synthetic data
        for col, scaler in self.dataset.scalers.items():
            data[[col]] = scaler.inverse_transform(data[[col]])

        # post-process
        data[self.dataset.categorical_features] = data[
            self.dataset.categorical_features
        ].astype(int)
        data[self.dataset.integer_features] = (
            data[self.dataset.integer_features].round(0).astype(int)
        )

        return data


# %%
