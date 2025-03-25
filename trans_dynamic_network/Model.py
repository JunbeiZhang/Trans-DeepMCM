# +--------------------------------------------------+
# | coding: utf-8                                    |
# | Author: Junbei Zhang                             |
# | Based on: Dynamic-DeepHit                        |
# +--------------------------------------------------+

import torch
import torch.nn as nn
import torch.nn.init as init
import math

_EPSILON = 1e-08  # Small constant to prevent division by zero

# Function to calculate sequence lengths
def get_seq_length(sequence):
    used = torch.sign(torch.max(torch.abs(sequence), 2)[0])  # Compute valid time steps
    tmp_length = torch.sum(used, 1).int()  # Sum valid time steps as sequence lengths
    return tmp_length

# Fully connected network definition
class FC_net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, activ_out, nlayers=2, activation='ReLU'):
        super(FC_net, self).__init__()
        if activation == 'ReLU':
            self.activ = nn.ReLU()
        elif activation == 'Tanh':
            self.activ = nn.Tanh()

        if activ_out == 'ReLU':
            self.activ_out = nn.ReLU()
        else:
            self.activ_out = activ_out

        self.dropout = dropout
        self.nlayers = nlayers
        self.FC_1 = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.FC_1.weight)
        self.FC_2 = nn.Linear(hidden_size, output_size)
        init.xavier_uniform_(self.FC_2.weight)

    def forward(self, x):
        for i in range(self.nlayers):
            if i != self.nlayers - 1:
                out0 = self.FC_1(x)
                out1 = self.activ(out0)
                out3 = nn.Dropout(p=self.dropout)(out1)
            else:
                out = self.FC_2(out3)
                if not self.activ_out == None:
                    out = self.activ_out(out)
        return out

# Positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Transformer model definition
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, input_dim, nhead):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=nhead,  # Use provided nhead
            dim_feedforward=hidden_size,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.input_size = input_size

        # Calculate size of all_last features
        size_all_last = 2 * (input_dim - 1)  # x_last and x_mi_last exclude the first column

        # Calculate input size for FC_net
        fc_input_size = input_size + size_all_last

        self.fc_net = FC_net(
            input_size=fc_input_size,
            output_size=1,
            hidden_size=hidden_size,
            dropout=dropout,
            activ_out=None,
            nlayers=2,
            activation='Tanh'
        )

    def forward(self, src, all_last):
        # src shape: [batch_size, seq_len, input_size]
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, input_size]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)  # [seq_len, batch_size, input_size]
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, input_size]

        seq_len = output.size(1)
        e_list = []
        for t in range(seq_len):
            tmp_h = output[:, t, :]  # [batch_size, input_size]
            # Calculate input for FC_net
            fc_input = torch.cat([tmp_h, all_last], dim=1)  # [batch_size, fc_input_size]
            e_ = self.fc_net(fc_input)
            e = torch.exp(e_)
            e_list.append(e)

        att_output = torch.stack(e_list, dim=1).squeeze(2)  # [batch_size, seq_len]

        return output, att_output

# Dynamic2StaticNet model with Transformer replacing RNN
class Dynamic2StaticNet(nn.Module):
    def __init__(self, input_dim, output_dim, network_settings, risks, optimizer='Adam'):
        super(Dynamic2StaticNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.risks = risks

        # Update parameter names to match new hyperparameter settings
        self.h_dim1 = network_settings['h_dim_Transformer']
        self.h_dim2 = network_settings['h_dim_FC']
        self.keep_prob = 1 - network_settings['keep_prob']
        self.num_Event = network_settings['num_Event']
        self.num_Category = network_settings['num_Category']
        self.nhead = network_settings['nhead']  # Retrieve nhead from network_settings

        # Replace RNN with Transformer; ensure num_layers is updated
        self.transformer = TransformerModel(
            input_size=input_dim * 2,
            hidden_size=self.h_dim1,
            num_layers=network_settings['num_layers_Transformer'],
            dropout=self.keep_prob,
            input_dim=self.input_dim,  # Pass input_dim
            nhead=self.nhead  # Pass nhead
        )

    def forward(self, x, x_mi):
        device = x.device

        # Get sequence lengths and create mask
        seq_length = get_seq_length(x).to(device)
        max_length = x.shape[1]
        feature_length = x.shape[2]
        tmp_range = torch.unsqueeze(torch.arange(0, max_length, 1), dim=0).to(device)
        rnn_mask2 = torch.eq(tmp_range, torch.unsqueeze(seq_length - 1, dim=1))

        x_last = torch.sum(torch.unsqueeze(rnn_mask2, dim=2).repeat(1, 1, feature_length) * x, dim=1)
        x_last = x_last[:, 1:]
        x_hist = x[:, :max_length - 1, :]

        x_mi_last = torch.sum(torch.unsqueeze(rnn_mask2, dim=2).repeat(1, 1, feature_length) * x_mi, dim=1)
        x_mi_last = x_mi_last[:, 1:]
        x_mi_hist = x_mi[:, :max_length - 1, :]

        all_last = torch.cat([x_last, x_mi_last], dim=1).to(device)
        all_hist = torch.cat([x_hist, x_mi_hist], dim=2).to(device)

        rnn_mask_att = torch.sum(x_hist, dim=2) != 0

        hidden, att_weight = self.transformer(all_hist, all_last)

        # Compute attention weights and normalize
        ej = att_weight.mul(rnn_mask_att)
        aj = torch.div(ej, torch.sum(ej, dim=1, keepdim=True) + _EPSILON)

        # Compute context vector
        attention = aj.unsqueeze(2).repeat(1, 1, hidden.size(2))
        c = torch.sum(attention.mul(hidden), dim=1)

        # Return context vector c
        return c
