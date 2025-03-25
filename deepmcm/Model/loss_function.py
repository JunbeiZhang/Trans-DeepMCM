# +--------------------------------------------------+
# | coding: utf-8                                    |
# | Author: JunbeiZhang                              |
# | Based on: DeepSurv(czifan)                       |
# +--------------------------------------------------+

import torch
import torch.nn as nn

class Regularization(object):
    def __init__(self, order, weight_decay):
        '''
        :param order: (int) Order of the regularization norm
        :param weight_decay: (float) Weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = float(weight_decay)

    def __call__(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss += torch.norm(w, p=self.order)
        reg_loss *= self.weight_decay
        return reg_loss

class CureRateLoss(nn.Module):
    '''
    Loss function for the Cure Rate Network
    '''
    def __init__(self, l2_reg):
        super(CureRateLoss, self).__init__()
        self.L2_reg = l2_reg
        self.reg = Regularization(order=2, weight_decay=self.L2_reg) if self.L2_reg > 0 else None
        self.bce_loss = nn.BCELoss()

    def forward(self, pi_pred, e, model):
        '''
        :param pi_pred: Predicted cure rate \( \pi \), tensor of shape [batch_size, 1]
        :param e: Event indicator, tensor of shape [batch_size, 1]
        :param model: Cure Rate Network model
        '''
        e = e.view(-1, 1)
        pi_pred = pi_pred.view(-1, 1)

        # Target value for cured status: 1 if cured, else 0
        # Since we cannot observe the true cured status, we approximate:
        # For events (e=1), not cured; for censored (e=0), status unknown
        # Handle censored data cautiously
        target = e  # Assume e=1 indicates not cured, e=0 indicates unknown

        # Predict the probability of not being cured as (1 - pi)
        prediction = (1 - pi_pred).clamp(min=1e-8, max=1 - 1e-8)  # Avoid log(0)

        # Compute binary cross-entropy loss
        loss = self.bce_loss(prediction, target)

        if self.reg is not None:
            loss += self.reg(model)

        return loss

class NegativeLogLikelihood(nn.Module):
    def __init__(self, config):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg) if self.L2_reg > 0 else None

    def forward(self, risk_pred, y, e, model):
        '''
        Negative log-likelihood loss function for the DeepSurv network
        '''
        y = y.view(-1)
        e = e.view(-1)
        risk_pred = risk_pred.view(-1)

        # Sort in descending order of time
        sorted_indices = torch.argsort(y, descending=True)
        y = y[sorted_indices]
        e = e[sorted_indices]
        risk_pred = risk_pred[sorted_indices]

        # Calculate cumulative risk
        exp_risk = torch.exp(risk_pred)
        cum_exp_risk = torch.cumsum(exp_risk, dim=0)
        log_cum_exp_risk = torch.log(cum_exp_risk + 1e-8)  # Avoid log(0)

        # Compute negative log-likelihood
        neg_log_likelihood = -torch.sum((risk_pred - log_cum_exp_risk) * e) / torch.sum(e)

        if self.reg is not None:
            neg_log_likelihood += self.reg(model)

        return neg_log_likelihood

class DeepMCMLoss(nn.Module):
    '''
    Total loss function for the DeepMCM model, combining Cure Rate loss and DeepSurv loss
    '''
    def __init__(self, cure_loss_config, surv_loss_config):
        super(DeepMCMLoss, self).__init__()
        self.cure_loss = CureRateLoss(cure_loss_config['l2_reg'])
        self.surv_loss = NegativeLogLikelihood(surv_loss_config)

    def forward(self, pi_pred, risk_pred, y, e, pi_model, surv_model):
        loss_cure = self.cure_loss(pi_pred, e, pi_model)  # Cure rate loss
        loss_surv = self.surv_loss(risk_pred, y, e, surv_model)  # DeepSurv loss
        total_loss = loss_cure + loss_surv  # Total loss is the sum of both
        return total_loss
