# DistVAE-Tabular

**DistVAE** is a novel approach to distributional learning in the VAE framework, focusing on accurately capturing the underlying distribution of the observed dataset through a nonparametric CDF estimation. 

We utilize the continuous ranked probability score (CRPS), a strictly proper scoring rule, as the reconstruction loss while preserving the mathematical derivation of the lower bound of the data log-likelihood. Additionally, we introduce a synthetic data generation mechanism that effectively preserves differential privacy.

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
distvae.generate_data # generate synthetic data
```
- See [example.ipynb](example.ipynb) for detailed example with `loan` dataset.
  - Link for download `loan` dataset: [https://www.kaggle.com/datasets/teertha/personal-loan-modeling](https://www.kaggle.com/datasets/teertha/personal-loan-modeling)

### Citation
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