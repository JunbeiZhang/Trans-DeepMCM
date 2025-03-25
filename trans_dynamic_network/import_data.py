# +--------------------------------------------------+
# | coding: utf-8                                    |
# | Author: Junbei Zhang                             |
# | Based on: Dynamic-DeepHit                        |
# +--------------------------------------------------+

import pandas as pd  # Used for data manipulation and analysis
import numpy as np  # Used for numerical computations
from torch.utils.data import Dataset  # Used to create custom PyTorch datasets
import torch  # Import the PyTorch library for deep learning

# Function: Normalize input data X based on the given normalization mode
def f_get_Normalization(X, norm_mode):
    num_Patient, num_Feature = np.shape(X)
    if norm_mode == 'standard':  # Standardization (zero mean, unit variance)
        for j in range(num_Feature):
            if np.nanstd(X[:, j]) != 0:
                X[:, j] = (X[:, j] - np.nanmean(X[:, j])) / np.nanstd(X[:, j])
            else:
                X[:, j] = (X[:, j] - np.nanmean(X[:, j]))
    elif norm_mode == 'normal':  # Min-Max normalization
        for j in range(num_Feature):
            X[:, j] = (X[:, j] - np.nanmin(X[:, j])) / (np.nanmax(X[:, j]) - np.nanmin(X[:, j]))
    else:
        print("Invalid normalization mode input!")
    return X

# Function: Generate mask1 for loss calculation (conditional probability)
def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''
        mask1 is used to obtain conditional probabilities (for calculating the denominator)
        The shape of mask1 is [N, num_Event, num_Category], with elements set to 1 before the last measurement time
    '''
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category])
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0] + 1)] = 1  # Mark up to the last valid measurement time
    return mask

# Function: Generate mask2 for loss calculation (log-likelihood loss)
def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
        mask2 is used to obtain log-likelihood loss
        The shape of mask2 is [N, num_Event, num_Category]
            If not censored: a certain element = 1 (others are 0)
            If censored: elements after the censoring time are set to 1 (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category])
    for i in range(np.shape(time)[0]):
        if label[i, 0] != 0:  # Not censored
            mask[i, int(label[i, 0] - 1), int(time[i, 0])] = 1
        else:  # label[i, 0] == 0: censored
            mask[i, :, int(time[i, 0] + 1):] = 1  # Fill with 1 from censoring time onwards
    return mask

# Function: Generate mask3 for loss calculation (ranking loss)
def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask3 is used to calculate ranking loss (for pairwise comparisons)
        The shape of mask3 is [N, num_Category].
        For longitudinal measurements:
        - Elements between the last measurement time and the event time (excluding and including, respectively) are set to 1
    '''
    mask = np.zeros([np.shape(time)[0], num_Category])
    for i in range(np.shape(time)[0]):
        t1 = int(meas_time[i, 0])  # Last measurement time
        t2 = int(time[i, 0])  # Censoring/event time
        mask[i, (t1 + 1):(t2 + 1)] = 1  # Exclude the last measurement time, include the event time
    return mask

# Function: Construct dataset from dataframe and feature list
def f_construct_dataset(df, feat_list):
    '''
        id   : Patient identifier
        tte  : Time to event or censoring
            - Must be synchronized based on reference time
        times: Measurement times of observations
            - Must be synchronized based on reference time (i.e., times start from 0)
        label: Event/censoring information
            - 0: Censored
            - 1: Event type 1
            - 2: Event type 2
    '''
    grouped = df.groupby(['id'])
    id_list = pd.unique(df['id'])
    max_meas = np.max(grouped.size())  # Maximum number of measurements per patient
    data = np.zeros([len(id_list), max_meas, len(feat_list) + 1])  # An extra column for delta times
    pat_info = np.zeros([len(id_list), 5])

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)
        pat_info[i, 4] = tmp.shape[0]  # Number of measurements
        pat_info[i, 3] = np.max(tmp['times'])  # Last measurement time
        pat_info[i, 2] = tmp['label'][0]  # Event type
        pat_info[i, 1] = tmp['tte'][0]  # Time to event
        pat_info[i, 0] = tmp['id'][0]

        # Fill the data matrix, first column is time difference, followed by features
        data[i, :int(pat_info[i, 4]), 1:] = tmp[feat_list]
        if pat_info[i, 4] > 1:
            data[i, :int(pat_info[i, 4] - 1), 0] = np.diff(tmp['times'])
        else:
            data[i, 0, 0] = tmp['times'][0]  # If only one measurement, time difference is the measurement time

    return pat_info, data

# Import and preprocess the dataset
def import_dataset(norm_mode='standard'):
    df_ = pd.read_csv(r'trans_dynamic_network\data\pbc2_filled.csv')  # Load dataset

    bin_list = ['drug', 'sex', 'ascites', 'hepatomegaly', 'spiders']
    cont_list = ['age', 'edema', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin', 'histologic']
    feat_list = cont_list + bin_list

    df_ = df_[['id', 'tte', 'times', 'label'] + feat_list]  # Keep only necessary columns
    df_org_ = df_.copy(deep=True)

    # Normalize continuous features
    df_[cont_list] = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

    # Construct normalized and original datasets
    pat_info, data = f_construct_dataset(df_, feat_list)
    _, data_org = f_construct_dataset(df_org_, feat_list)

    data_mi = np.zeros(np.shape(data))
    data_mi[np.isnan(data)] = 1
    data_org[np.isnan(data)] = 0
    data[np.isnan(data)] = 0

    x_dim = np.shape(data)[2]
    x_dim_cont = len(cont_list)
    x_dim_bin = len(bin_list)

    last_meas = pat_info[:, [3]]  # Last measurement time
    label = pat_info[:, [2]]  # Event type
    time = pat_info[:, [1]]  # Time to event

    num_Category = int(np.max(pat_info[:, 1]) * 1.2)
    num_Event = len(np.unique(label)) - 1

    if num_Event == 1:
        label[np.where(label != 0)] = 1  # Single-risk scenario

    mask1 = f_get_fc_mask1(last_meas, num_Event, num_Category)
    mask2 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask3 = f_get_fc_mask3(time, last_meas, num_Category)
    
    # Output the values of the first sample's first column at all time steps
    DIM = (x_dim, x_dim_cont, x_dim_bin)
    DATA = (data, time, label)
    MASK = (mask1, mask2, mask3)

    return DIM, DATA, MASK, data_mi

# Import and preprocess the simulated dataset
def import_dataset_other(norm_mode='standard'):
    df_ = pd.read_csv(r"trans_dynamic_network\data\data2.csv")  # Load simulated dataset

    # Define binary features and continuous features
    bin_list = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']  # Binary features
    cont_list = ['y']  # Continuous features
    
    # df_ = pd.read_csv(r"trans_dynamic_network\data\aids_filled.csv")  # Load simulated dataset

    # # Define binary features and continuous features
    # bin_list = ['sex']
    # cont_list = ['gfr', 'weight', 'age']

    feat_list = cont_list + bin_list

    df_ = df_[['id', 'tte', 'times', 'label'] + feat_list]  # Keep only necessary columns
    df_org_ = df_.copy(deep=True)

    # Normalize continuous features
    df_[cont_list] = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

    # Construct normalized and original datasets
    pat_info, data = f_construct_dataset(df_, feat_list)
    _, data_org = f_construct_dataset(df_org_, feat_list)

    data_mi = np.zeros(np.shape(data))
    data_mi[np.isnan(data)] = 1
    data_org[np.isnan(data)] = 0
    data[np.isnan(data)] = 0

    x_dim = np.shape(data)[2]
    x_dim_cont = len(cont_list)
    x_dim_bin = len(bin_list)

    last_meas = pat_info[:, [3]]  # Last measurement time
    label = pat_info[:, [2]]  # Event type
    time = pat_info[:, [1]]  # Time to event

    num_Category = int(np.max(pat_info[:, 1]) * 1.2)
    num_Event = len(np.unique(label)) - 1

    if num_Event == 1:
        label[np.where(label != 0)] = 1  # Single-risk scenario

    mask1 = f_get_fc_mask1(last_meas, num_Event, num_Category)
    mask2 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask3 = f_get_fc_mask3(time, last_meas, num_Category)

    DIM = (x_dim, x_dim_cont, x_dim_bin)
    DATA = (data, time, label)
    MASK = (mask1, mask2, mask3)

    return DIM, DATA, MASK, data_mi


# Custom dataset class for PyTorch DataLoader
class Dataset_all(Dataset):
    def __init__(self, d1, d2, d3, d4, d5, d6, d7):
        self.d1 = torch.tensor(d1, dtype=torch.float32)  # Data
        self.d2 = torch.tensor(d2, dtype=torch.float32)  # Missing data indicator
        self.d3 = torch.tensor(d3, dtype=torch.float32)  # Time to event
        self.d4 = torch.tensor(d4, dtype=torch.float32)  # Event labels
        self.d5 = torch.tensor(d5, dtype=torch.float32)  # Mask1
        self.d6 = torch.tensor(d6, dtype=torch.float32)  # Mask2
        self.d7 = torch.tensor(d7, dtype=torch.float32)  # Mask3

    def __len__(self):
        return self.d1.shape[0]  # Number of samples

    def __getitem__(self, item):
        return self.d1[item, :, :], self.d2[item, :, :], self.d3[item, :], \
               self.d4[item, :], self.d5[item, :, :], self.d6[item, :, :], self.d7[item, :]
