# coding=utf-8
import numpy as np
import torch.nn.functional as F
import torch
from datautil.util import random_pairs_of_minibatches
from alg.algs.ERM import ERM


class Mixup1(ERM):
    def __init__(self, args):
        super(Mixup1, self).__init__(args)
        self.args = args

    def update(self, minibatches, opt, sch):
        objective = 0

        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches(self.args, minibatches):
            lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)

            x = (lam * xi + (1 - lam) * xj).cuda().float()
            batch_size = self.args.batch_size
            # Transform label to one-hot
            yi = torch.zeros(batch_size, self.args.num_classes).scatter_(1, yi.view(-1, 1).long(), 1)
            yj = torch.zeros(batch_size, self.args.num_classes).scatter_(1, yj.view(-1, 1).long(), 1)
            y = (lam * yi + (1 - lam) * yj).cuda().float()

            predictions = self.predict(x)

            objective +=  -torch.mean(torch.sum(F.log_softmax(predictions, dim=1) * y, dim=1))

            # objective += lam * F.cross_entropy(predictions, yi.cuda().long())
            # objective += (1 - lam) * \
            #     F.cross_entropy(predictions, yj.cuda().long())

        objective /= len(minibatches)

        opt.zero_grad()
        objective.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': objective.item()}
