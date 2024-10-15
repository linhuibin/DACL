# coding=utf-8
import torch
import copy
import torch.nn.functional as F

from alg.opt import *
import torch.autograd as autograd
from datautil.util import random_pairs_of_minibatches_by_domainperm, split_meta_train_test
from alg.algs.ERM import ERM


class MLDG(ERM):
    def __init__(self, args):
        super(MLDG, self).__init__(args)
        self.args = args

    def update(self, minibatches, opt, sch):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)
            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)
            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)
        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        opt.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches):

            xi, yi, xj, yj = xi.cuda().float(), yi.cuda(
            ).long(), xj.cuda().float(), yj.cuda().long()
            inner_net = copy.deepcopy(self.network)  # inner_net inner_net is used to store tmp gradient values

            inner_opt = get_optimizer(inner_net, self.args, True)
            inner_sch = get_scheduler(inner_opt, self.args)

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()  # Θ' = Θ - lr * ▽Θ    F() minimise
            if inner_sch:
                inner_sch.step()

            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)       # network parameter = Θ'

            objective += inner_obj.item()    # F loss

            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),  # G() minimise
                                         allow_unused=True)

            objective += (self.args.mldg_beta * loss_inner_j).item()  # G loss

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.args.mldg_beta * g_j.data / num_mb)
            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)  # 类似 MixUp

        opt.step()
        if sch:
            sch.step()
        return {'total': objective}
