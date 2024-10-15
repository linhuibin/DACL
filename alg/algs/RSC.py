# coding=utf-8
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

from alg.algs.ERM import ERM


class RSC(ERM):
    def __init__(self, args):
        super(RSC, self).__init__(args)
        self.drop_f = (1 - args.rsc_f_drop_factor) * 100
        self.drop_b = (1 - args.rsc_b_drop_factor) * 100
        self.num_classes = args.num_classes

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        all_f = self.featurizer(all_x)
        all_p = self.classifier(all_f)  #
        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]   # loss 与 后面 backforward不是同一个更新的图，参数不同，这里更新的是对特征f求导，所以不用retain_graph
        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()  # 前百分drop_f的数进行遮罩，梯度大的主特征被遮罩  lt <

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)   # [b] 根据原论文验证changes最终是否会收敛  ??? test Does the loss before and after masking decrease?
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)  # 改变最大的前top drop_b的丢弃，改变不大的保留
        mask = torch.logical_or(mask_f, mask_b).float()   # batch变化大且梯度大的特征才会被mask为0

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)    # All logits are calculated together, cross_entropy automatically averages all logits, no need to divide by number of minibatches
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        return {'class': loss.item()}
