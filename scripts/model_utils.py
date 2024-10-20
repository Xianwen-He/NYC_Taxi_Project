# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:24:04 2024

@author: XianwenHe
"""

import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder

class NYCTaxiExampleDataset(torch.utils.data.Dataset):
    """Trainin data object for our nyc taxi data"""
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Initialize the dataset
        """
        self.X_train = X_train
        self.y_train = y_train
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.X = torch.from_numpy(self._one_hot_X().toarray()) # potentially smarter ways to deal with sparse here
        self.y = torch.from_numpy(self.y_train.values)
        self.X_enc_shape = self.X.shape[-1]
        print(f"encoded shape is {self.X_enc_shape}")
    
    def __len__(self):
        """
        Return the size of the dataset
        """
        return len(self.X)

    def __getitem__(self, i):
        """
        Return the ith instance of the dataset
        """
        return self.X[i], self.y[i]
        
    def _one_hot_X(self):
        """
        Encode categorical features in the dataset as a one-hot numeric array
        """
        return self.one_hot_encoder.fit_transform(self.X_train)


class MLP(nn.Module):
    """Multilayer Perceptron for regression. """
    def __init__(self, encoded_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoded_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
    
    def forward(self, x):
        return self.layers(x)