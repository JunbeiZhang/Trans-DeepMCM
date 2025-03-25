# +--------------------------------------------------+
# | coding: utf-8                                    |
# | Author: JunbeiZhang                              |
# | Based on: DeepSurv(czifan)                       |
# +--------------------------------------------------+

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class CureRateNetwork(nn.Module):
    '''
    Cure Rate Network uses a Deep Neural Network (DNN) to estimate the cure rate pi(x).
    '''
    def __init__(self, config, device):
        super(CureRateNetwork, self).__init__()
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        self.device = device

        # Parse network configuration
        self.dims = [int(d) for d in config['dims']]
        self.drop = float(self.drop) if self.drop else None
        self.norm = str(self.norm).lower() in ('true', '1', 't')
        self.activation = self.activation.strip("'\"")
        self.model = self._build_network()
        self.model.to(self.device)

    def _build_network(self):
        layers = []
        for i in range(len(self.dims)-1):
            layer = nn.Linear(self.dims[i], self.dims[i+1]).to(self.device)  # Add linear layer
            if self.drop is not None and i != 0:  # Add Dropout after non-input layers
                layers.append(nn.Dropout(self.drop))
            layers.append(layer)
            if self.norm:  # Add batch normalization layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            if i != len(self.dims)-2:  # Add activation function before the last layer
                layers.append(getattr(nn, self.activation)())
            # No activation function before the last layer, use Sigmoid to output pi
        return nn.Sequential(*layers)

    def forward(self, X):
        logits = X
        for layer in self.model:
            if isinstance(layer, nn.BatchNorm1d) and logits.shape[0] == 1:
                continue  # Skip BatchNorm layer
            logits = layer(logits)
        pi = torch.sigmoid(- logits)  # Use Sigmoid activation function to calculate cure rate
        return pi

class DeepSurv(nn.Module):
    ''' DeepSurv network for risk prediction of uncured groups. '''
    def __init__(self, config, device):
        super(DeepSurv, self).__init__()
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        self.device = device

        # Parse network configuration
        self.dims = [int(d) for d in config['dims']]
        self.drop = float(self.drop) if self.drop else None
        self.norm = str(self.norm).lower() in ('true', '1', 't')
        self.activation = self.activation.strip("'\"")
        self.model = self._build_network()
        self.model.to(self.device)

    def _build_network(self):
        layers = []
        for i in range(len(self.dims)-1):
            layer = nn.Linear(self.dims[i], self.dims[i+1]).to(self.device)  # Add linear layer
            if self.drop is not None and i != 0:  # Add Dropout after non-input layers
                layers.append(nn.Dropout(self.drop))
            layers.append(layer)
            if self.norm:  # Add batch normalization layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            layers.append(getattr(nn, self.activation)())  # Add activation function
        return nn.Sequential(*layers)

    def forward(self, X):
        logits = X
        for layer in self.model:
            if isinstance(layer, nn.BatchNorm1d) and logits.shape[0] == 1:
                continue  # Skip BatchNorm layer
            logits = layer(logits)
        return logits

class DeepMCM(nn.Module):
    '''
    Deep Mixture Cure Model integrates the Cure Rate Network and DeepSurv network.
    '''
    def __init__(self, pi_config, surv_config, device, cure_threshold=0.8):
        super(DeepMCM, self).__init__()
        self.device = device
        self.cure_threshold = cure_threshold  # Cure threshold
        self.pi_network = CureRateNetwork(pi_config, device)  # Initialize Cure Rate Network
        self.surv_network = DeepSurv(surv_config, device)  # Initialize DeepSurv Network

    def forward(self, X):
        pi = self.pi_network(X)  # Get cure rate prediction, shape [batch_size, 1]

        # Determine uncured samples
        uncured_mask = (pi < self.cure_threshold)  # Boolean tensor of shape [batch_size, 1]

        # Initialize risk_u, shape [batch_size, 1]
        risk_u = torch.zeros_like(pi)

        # If there are uncured samples, compute corresponding risk_u
        if uncured_mask.any():
            # Flatten boolean mask and use `.nonzero(as_tuple=True)` to get indices of uncured samples
            uncured_indices = uncured_mask.squeeze().nonzero(as_tuple=True)[0]

            # Extract features of uncured samples
            X_uncured = X[uncured_indices]
            # Compute risk predictions for uncured samples, output shape [n_uncured, 1]
            risk_u_uncured = self.surv_network(X_uncured)

            # Use indices to fill back the risk predictions into risk_u at the corresponding positions
            risk_u[uncured_indices] = risk_u_uncured

        # Calculate survival function S_u(t) for the uncured group
        S_u = torch.exp(-risk_u)  # Shape [batch_size, 1]

        # Calculate overall survival function S_pop(t)
        S_pop = pi + (1 - pi) * S_u  # Shape [batch_size, 1]

        # Calculate overall hazard function h_pop(t)
        numerator = (1 - pi) * risk_u * S_u  # Shape [batch_size, 1]
        h_pop = numerator / S_pop  # Shape [batch_size, 1]

        # Return cure rate pi and overall risk prediction h_pop
        return pi, h_pop
