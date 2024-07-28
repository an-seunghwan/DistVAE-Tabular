#%%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
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
data = data[continuous_features + categorical_features]
ClfTarget = "Personal Loan"
#%%
from distvae import DistVAE
distvae = DistVAE(
    data,
    continuous_features,
    categorical_features,
    integer_features,
    ClfTarget,
    epochs=5 # for quick checking
)
#%%
distvae.train()
#%%
syndata = distvae.generate_data(100)
syndata
#%%