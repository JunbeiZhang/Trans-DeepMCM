# +--------------------------------------------------+
# | coding: utf-8                                    |
# | Author: JunbeiZhang                              |
# | Based on: DeepSurv(czifan)                       |
# +--------------------------------------------------+

import torch

class Regularization(object):
    def __init__(self, order, weight_decay):
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = float(weight_decay)

    def __call__(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss
