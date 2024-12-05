# DistVAE-Tabular

**DistVAE** is a novel approach to distributional learning in the VAE framework, focusing on accurately capturing the underlying distribution of the observed dataset through a nonparametric quantile estimation.

We utilize the continuous ranked probability score (CRPS), a strictly proper scoring rule, as the reconstruction loss while preserving the mathematical derivation of the lower bound of the data log-likelihood. Additionally, we introduce a synthetic data generation mechanism that effectively preserves differential privacy.

> For a detailed method explanations, check our paper! [(link)](https://openreview.net/pdf?id=GxL6PrmEUw)

### 1. Installation
Install using pip:
```
pip install distvae-tabular
```

### 2. Usage
```python
from distvae_tabular import distvae
```
```python
distvae.DistVAE # DistVAE model
distvae.generate_data # function for generating synthetic dataset
```
- See [example.ipynb](example.ipynb) for detailed example and its results with `loan` dataset.
  - Link for download `loan` dataset: [https://www.kaggle.com/datasets/teertha/personal-loan-modeling](https://www.kaggle.com/datasets/teertha/personal-loan-modeling)

#### Example
```python
"""device setting"""
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""load dataset and specify column types"""
import pandas as pd
data = pd.read_csv('./loan.csv')
continuous_features = [
    'Age',
    'Experience',
    'Income',
    'CCAvg',
    'Mortgage',
]
categorical_features = [
    'Family',
    'Personal Loan',
    'Securities Account',
    'CD Account',
    'Online',
    'CreditCard'
]
integer_features = [
    'Age',
    'Experience',
    'Income',
    'Mortgage'
]

"""DistVAE"""
from distvae_tabular import distvae

distvae = distvae.DistVAE(
    data=data, # the observed tabular dataset
    continuous_features=continuous_features, # the list of continuous columns of data
    categorical_features=categorical_features, # the list of categorical columns of data
    integer_features=integer_features, # the list of integer-type columns of data

    seed=42, # seed for repeatable results
    latent_dim=4, # the latent dimension size
    beta=0.1, # scale parameter of asymmetric Laplace distribution
    hidden_dim=128, # the number of nodes in MLP

    epochs=5, # the number of epochs (for quick checking)
    batch_size=256, # the batch size
    lr=0.001, # learning rate

    step=0.1, # interval size between knots
    threshold=1e-8, # threshold for clipping alpha_tild (numerical stability)
    device="cpu"
)

"""training"""
distvae.train()

"""generate synthetic data"""
syndata = distvae.generate_data(100)
syndata

"""generate synthetic data with Differential Privacy"""
syndata = distvae.generate_data(100, lambda_=0.1)
syndata
```

### 3. Citation
If you use this code or package, please cite our associated paper:
```
@article{an2024distributional,
  title={Distributional learning of variational AutoEncoder: application to synthetic data generation},
  author={An, Seunghwan and Jeon, Jong-June},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
