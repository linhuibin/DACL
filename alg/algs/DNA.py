# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from utils.util import PJS_loss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

class DNA(Algorithm):
    """
    Diversified Neural Averaging(DNA)
    """

    def __init__(self, args):
        super(DNA, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.MCdropClassifier(
            in_features=self.featurizer.in_features,
            num_classes=args.num_classes,
            bottleneck_dim=1024,
            dropout_rate=0.5,
            dropout_type='Bernoulli'
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.train_sample_num = 5
        self.lambda_v = 0.1
        self.amp = args.amp
        self.scaler = GradScaler(enabled=args.amp)

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        with autocast(enabled=self.amp):
            all_f = self.featurizer(all_x)
            loss_pjs = 0.0
            row_index = torch.arange(0, all_x.size(0))

            probs_y = []
            for i in range(self.train_sample_num):
                pred = self.classifier(all_f)
                prob = F.softmax(pred, dim=1)
                prob_y = prob[row_index, all_y]
                probs_y.append(prob_y.unsqueeze(0))
                loss_pjs += PJS_loss(prob, all_y)

            probs_y = torch.cat(probs_y, dim=0)
            X = torch.sqrt(torch.log(2 / (1 + probs_y)) + probs_y * torch.log(2 * probs_y / (1 + probs_y)) + 1e-6)
            loss_v = (X.pow(2).mean(dim=0) - X.mean(dim=0).pow(2)).mean()
            loss_pjs /= self.train_sample_num
            loss = loss_pjs - self.lambda_v * loss_v

        opt.zero_grad()
        self.scaler.scale(loss).backward()  # 为了梯度放大
        self.scaler.step(
            opt)  # 首先把梯度值unscale回来，如果梯度值不是inf或NaN,则调用optimizer.step()来更新权重，否则，忽略step调用，从而保证权重不更新。　　 scaler.step(optimizer)
        self.scaler.update()  # 准备着，看是否要增大scaler

        if sch:
            sch.step()

        return {"total": loss.item(), "loss_c": loss_pjs.item(), "loss_v": loss_v.item()}

    def predict(self, x):
        return self.network(x)

