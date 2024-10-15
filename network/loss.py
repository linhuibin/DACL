"""
Author: Huibin Lin (huibinlin@outlook.com)
Date: March 20, 2023
"""
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class ProtoLoss(nn.Module):
    """Prototypes Contrastive Learning"""

    def __init__(self, temperature=0.1, num_classes=None, num_domains=None):
        super(ProtoLoss, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.num_domains = num_domains

    def forward(self, feature, all_feature, labels=None, all_lables=None, dl=None, all_dl=None,
                score=None):
        """Compute loss for model.
        Args:
            fea: hidden vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
            proto_fea: hidden vector of shape [bp, ...].
            proto_lables: target label of prototypes shape [bp]
            mask: contrastive mask of shape [bsz, bp], mask_{i,j}=1 if sample j
                has the same class as sample i.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if feature.is_cuda
                  else torch.device('cpu'))
        fea = nn.functional.normalize(feature, dim=-1)
        all_fea = nn.functional.normalize(all_feature, dim=-1)
        if dl != None:
            d_mask = score[dl, :][:, all_dl].float().detach()

        density = self.temperature
        batch_size = fea.shape[0]
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        labels_o = torch.nn.functional.one_hot(labels, self.num_classes).float().cuda()  # one-hot label
        # if dl != None:
        #     dl_o = torch.nn.functional.one_hot(dl, self.num_domains).float().cuda()  # one-hot domain label

        bp = all_fea.shape[0]
        if all_lables.shape[0] != bp:
            raise ValueError('Num of prototype labels does not match num of prototype features')

        all_lables_o = torch.nn.functional.one_hot(all_lables, self.num_classes).float().cuda()  # one-hot label

        mask = torch.matmul(labels_o, all_lables_o.T).float().to(device)  # [bsz,bp]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(fea, all_fea.T),
            density)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        # 分母
        if dl != None:
            log_prob =  logits - torch.log((exp_logits).sum(1, keepdim=True))
            mean_log_prob_pos = (mask * d_mask * log_prob).sum(1) / (mask).sum(1)
        else:
            # compute log_prob
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask *  log_prob).sum(1) / (mask).sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss
