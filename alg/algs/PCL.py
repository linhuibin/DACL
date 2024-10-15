import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from alg.algs.base import Algorithm
from alg.modelopera import get_fea
from network import common_network
from network.pcl_losses import ProxyPLoss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler


class PCL(Algorithm):

    def __init__(self, args):
        super(PCL, self).__init__(args)
        self.args = args

        self.encoder, self.scale, self.pcl_weights = common_network.encoder(args)
        self._initialize_weights(self.encoder)
        self.fea_proj, self.fc_proj = common_network.fea_proj(args)
        nn.init.kaiming_uniform_(self.fc_proj, mode='fan_out', a=math.sqrt(5))
        self.featurizer = get_fea(args)
        self.classifier = nn.Parameter(torch.FloatTensor(args.num_classes,
                                                         self.fea_proj[0].out_features))
        nn.init.kaiming_uniform_(self.classifier, mode='fan_out', a=math.sqrt(5))

        self.proxycloss = ProxyPLoss(num_classes=args.num_classes, scale=self.scale)

        self.amp = args.amp
        self.scaler = GradScaler(enabled=args.amp)

    def _initialize_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        with autocast(enabled=self.amp):
            rep, pred = self.predict_val(all_x)   # all_x --> all_z --> all_e
            loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), all_y)

            fc_proj = F.linear(self.classifier, self.fc_proj)    # w --> v
            assert fc_proj.requires_grad == True
            loss_pcl = self.proxycloss(rep, all_y, fc_proj)

            loss = loss_cls + self.pcl_weights * loss_pcl

        opt.zero_grad()
        self.scaler.scale(loss).backward()  # 为了梯度放大
        self.scaler.step(
            opt)  # 首先把梯度值unscale回来，如果梯度值不是inf或NaN,则调用optimizer.step()来更新权重，否则，忽略step调用，从而保证权重不更新。　　 scaler.step(optimizer)
        self.scaler.update()  # 准备着，看是否要增大scaler

        if sch:
            sch.step()

        return {"total": loss.item(), "loss_cls": loss_cls.item(), "loss_pcl": loss_pcl.item()}

    def predict_val(self, x):
        x = self.featurizer(x)
        x = self.encoder(x)
        rep = self.fea_proj(x)
        pred = F.linear(x, self.classifier)

        return rep, pred

    def predict(self, x):
        x = self.featurizer(x)
        x = self.encoder(x)
        pred = F.linear(x, self.classifier)

        return pred
