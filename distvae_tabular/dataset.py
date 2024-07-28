#%%
import pandas as pd
import numpy as np
from collections import namedtuple
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['num_features', 'num_continuous_features', 'num_categories'])
#%%
class CustomDataset(Dataset):
    def __init__(
        self, 
        data,
        continuous_features=[], 
        categorical_features=[], 
        integer_features=[], 
        ClfTarget=None,
        ):
        
        if len(continuous_features) == 0:
            continuous_features = data.columns
        
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.ClfTarget = ClfTarget
        
        self.features = self.continuous_features + self.categorical_features
        self.col_2_idx = {col : i for i, col in enumerate(data[self.features].columns.to_list())}
        self.num_continuous_features = len(self.continuous_features)
        
        # encoding categorical dataset
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes)
        self.num_categories = data[self.categorical_features].nunique(axis=0).to_list()

        data = data[self.features] # select features for training
        data = pd.get_dummies(
            data, columns=self.categorical_features, prefix_sep="###"
        )
        data = data.reset_index(drop=True)
        self.raw_data = data
        
        self.scalers = {}
        transformed = []
        for continuous_feature in tqdm(self.continuous_features, desc="Tranform Continuous Features..."):
            transformed.append(self.transform_continuous(data, continuous_feature))
        
        self.data = np.concatenate(
            transformed + [
                data[[x for x in data.columns if x not in self.continuous_features]].values
            ], axis=1
        )
        
        self.EncodedInfo = EncodedInfo(
            len(self.features), self.num_continuous_features, self.num_categories)
        
        raw_undummified = self.undummify(
            self.raw_data[[x for x in self.raw_data.columns if x not in self.continuous_features]]
        )
        self.raw_data = pd.concat(
            [self.raw_data[self.continuous_features], raw_undummified],
            axis=1
        )
        
    def transform_continuous(self, data, col):
        feature = data[[col]].to_numpy().astype(float)
        scaler = StandardScaler().fit(feature)
        self.scalers[col] = scaler
        return scaler.transform(feature)
    
    def undummify(self, df, prefix_sep="###"):
        cols2collapse = {
            item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
        }
        series_list = []
        for col, needs_to_collapse in cols2collapse.items():
            if needs_to_collapse:
                undummified = (
                    df.filter(like=col)
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                    .rename(col)
                )
                series_list.append(undummified.astype(int))
            else:
                series_list.append(df[col])
        undummified_df = pd.concat(series_list, axis=1)
        return undummified_df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%