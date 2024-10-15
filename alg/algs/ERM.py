# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERM, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier)

        self.network = nn.Sequential(
            self.featurizer, self.classifier)

        self.amp = args.amp
        self.scaler = GradScaler(enabled=args.amp)

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        with autocast(enabled=self.amp):
            loss = F.cross_entropy(self.predict(all_x), all_y)

        opt.zero_grad()
        self.scaler.scale(loss).backward()  # 为了梯度放大
        self.scaler.step(
            opt)  # 首先把梯度值unscale回来，如果梯度值不是inf或NaN,则调用optimizer.step()来更新权重，否则，忽略step调用，从而保证权重不更新。　　 scaler.step(optimizer)
        self.scaler.update()  # 准备着，看是否要增大scaler

        if sch:
            sch.step()
        return {'class': loss.item()}

    def predict(self, x):
        return self.network(x)
