# coding=utf-8
import numpy as np
import torch.nn.functional as F

from datautil.util import random_pairs_of_minibatches
from alg.algs.ERM import ERM
from torch.cuda.amp import autocast as autocast

class Mixup(ERM):
    def __init__(self, args):
        super(Mixup, self).__init__(args)
        self.args = args

    def update(self, minibatches, opt, sch):
        objective = 0

        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches(self.args, minibatches):
            lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)

            x = (lam * xi + (1 - lam) * xj).cuda().float()

            with autocast(enabled=self.amp):
                predictions = self.predict(x)

                objective += lam * F.cross_entropy(predictions, yi.cuda().long())
                objective += (1 - lam) * \
                    F.cross_entropy(predictions, yj.cuda().long())

        objective /= len(minibatches)

        opt.zero_grad()
        self.scaler.scale(objective).backward()  # 为了梯度放大
        self.scaler.step(
            opt)  # 首先把梯度值unscale回来，如果梯度值不是inf或NaN,则调用optimizer.step()来更新权重，否则，忽略step调用，从而保证权重不更新。　　 scaler.step(optimizer)
        self.scaler.update()  # 准备着，看是否要增大scaler

        if sch:
            sch.step()
        return {'class': objective.item()}