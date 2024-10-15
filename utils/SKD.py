import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss
    

def CS_KD(net, inputs, targets):
    loss_div = torch.tensor(0.).cuda()
    loss_cls = torch.tensor(0.).cuda()

    batch_size = inputs.size(0)
    targets = targets[:batch_size//2]
    logit, _ = net(inputs[:batch_size//2])
    with torch.no_grad():
        outputs_cls, _ = net(inputs[batch_size//2:])
    loss_cls += nn.CrossEntropyLoss()(logit, targets)
    loss_div += DistillKL(3.0)(logit, outputs_cls.detach())

    return loss_cls+loss_div



def BYOT(net, inputs, targets):
    loss_div = torch.tensor(0.).cuda()
    loss_cls = torch.tensor(0.).cuda()
    logits, features = net(inputs)
    loss_cls = nn.CrossEntropyLoss()(logits, targets)
    
    for i in range(1, len(features)):
        if i != 1:
            loss_div += 0.5 * 0.1 * ((features[i] - features[1].detach()) ** 2).mean()
    
    # logit = logits[0]   
    return loss_cls+loss_div


def TF_KD_reg(outputs, targets, num_classes=100, epsilon=0.1, T=20):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes),
                                 fill_value=epsilon / (num_classes - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1),
                             value=1-epsilon)
    log_prob = F.log_softmax(outputs / T, dim=1)
    loss = - torch.sum(log_prob * smoothed_labels/ T) / N
    return loss




