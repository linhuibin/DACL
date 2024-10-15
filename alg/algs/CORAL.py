# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM
from torch.cuda.amp import autocast as autocast

class CORAL(ERM):
    def __init__(self, args):
        super(CORAL, self).__init__(args)
        self.args = args
        self.kernel_type = "mean_cov"

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)  # i-th feature mean (batch norm)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)    # 行和列相乘，维度与维度做乘积累加除以 n-1即协方差
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()   # Remove this item the performance sometimes may improve more
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, minibatches, opt, sch):
        objective = 0
        penalty = 0
        nmb = len(minibatches)   # train domains

        with autocast(enabled=self.amp):
            features = [self.featurizer(
                data[0].cuda().float()) for data in minibatches]
            classifs = [self.classifier(fi) for fi in features]
            targets = [data[1].cuda().long() for data in minibatches]

            for i in range(nmb):      # Perform corss_entropy on logits by a minibatches (domain), you need to continue to divide by nmb
                objective += F.cross_entropy(classifs[i], targets[i])
                for j in range(i + 1, nmb):
                    penalty += self.coral(features[i], features[j])

            objective /= nmb
            if nmb > 1:
                penalty /= (nmb * (nmb - 1) / 2)

        opt.zero_grad()
        self.scaler.scale((objective + (self.args.mmd_gamma*penalty))).backward()  # 为了梯度放大
        self.scaler.step(
            opt)  # 首先把梯度值unscale回来，如果梯度值不是inf或NaN,则调用optimizer.step()来更新权重，否则，忽略step调用，从而保证权重不更新。　　 scaler.step(optimizer)
        self.scaler.update()  # 准备着，看是否要增大scaler

        if sch:
            sch.step()
        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'class': objective.item(), 'coral': penalty, 'total': (objective.item() + (self.args.mmd_gamma*penalty))}
