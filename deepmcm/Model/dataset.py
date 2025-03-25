# +--------------------------------------------------+
# | coding: utf-8                                    |
# | Author: JunbeiZhang                              |
# | Based on: DeepSurv(czifan)                       |
# +--------------------------------------------------+

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset
import numpy as np

class SurvivalDataset(Dataset):
    ''' Dataset class for the DeepMCM model '''
    def __init__(self, X, y, e):
        ''' Initialize with data arrays

        :param X: (np.array) (n_samples, n_features)
            Feature matrix
        :param y: (np.array) (n_samples, 1)
            Survival times
        :param e: (np.array) (n_samples, 1)
            Event indicators (1: event occurred; 0: censored)
        '''
        self.X = X
        self.y = y
        self.e = e
        # Normalize the data
        self._normalize()
        print('=> Loaded {} samples'.format(self.X.shape[0]))

    def _normalize(self):
        ''' Normalize the X data '''
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
    
    def __getitem__(self, index):
        ''' Get a specific data item '''
        X_item = self.X[index]
        y_item = self.y[index]
        e_item = self.e[index]
        # Convert to torch tensors
        X_tensor = torch.from_numpy(X_item).float()
        y_tensor = torch.from_numpy(y_item).float()
        e_tensor = torch.from_numpy(e_item).float()
        return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.X.shape[0]
