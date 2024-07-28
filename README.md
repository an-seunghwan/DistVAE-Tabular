# DistVAE-Tabular

### 1. Install
```
pip install distvae-tabular
```

### 2. Usage
```
from distvae_tabular import distvae

distvae.DistVAE # DistVAE model
distvae.generate_data # generate synthetic data
```
- See `example.ipynb` for detailed usage with `loan` dataset.
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