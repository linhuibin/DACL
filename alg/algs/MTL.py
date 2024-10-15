# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm

class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, args):
        super(MTL, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features * 2, args.classifier)

        self.register_buffer('embeddings',
                             torch.zeros(args.domain_num - len(args.test_envs),
                                         self.featurizer.in_features))
        self.ema = args.mtl_ema
        self.args = args

    def update(self, minibatches, opt, sch):
        loss = 0
        for env, data in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(data[0].cuda().float(), env), data[1].cuda().long())
        # loss /= len(minibatches)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()

        return self.classifier(torch.cat((features, embedding), 1))
