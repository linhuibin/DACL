import torch
import torch.nn.functional as F
from datautil.util import random_pairs_of_minibatches_by_domainperm, colorful_spectrum_mix
from utils.util import get_current_consistency_weight, interleave, de_interleave
from network.ema import ModelEMA
from alg.algs.ERM import ERM
from torch.cuda.amp import autocast as autocast


class FACT(ERM):
    def __init__(self, args):
        super(FACT, self).__init__(args)
        self.args = args
        self.teacher_network = ModelEMA(self.network, args)
        self.update_count = 0
        self.weight = args.weight
        self.uniform = args.uniform

    def update(self, minibatches, opt, sch):
        total_loss = 0
        current_epoch = self.update_count / 100
        for (xi, yi), (xj, yj) in random_pairs_of_minibatches_by_domainperm(minibatches):
            xi = xi.cuda()
            xj = xj.cuda()
            xi_mix, xj_mix = colorful_spectrum_mix(xi, xj, uniform=self.uniform)
            # inputs = interleave(torch.cat((xi, xj, xi_mix, xj_mix), dim=0), 4)
            inputs = torch.cat((xi, xj, xi_mix, xj_mix), dim=0)
            labels = torch.cat((yi, yj), dim=0).cuda()

            # import matplotlib.pyplot as plt
            # xp = xj_mix[0].cpu().numpy().transpose(1, 2, 0)
            # plt.imshow(xp)
            # plt.show()

            with autocast(enabled=self.amp):
                logits = self.network(inputs)
                # logits = de_interleave(logits, 4)
                logits_ori, logits_aug = torch.split(logits, [2 * self.args.batch_size, 2 * self.args.batch_size],
                                                     dim=0)

                with torch.no_grad():
                    logits_teacher = self.teacher_network.ema(inputs)
                    # logits_teacher = de_interleave(logits_teacher, 4)
                    logits_ori_tea, logits_aug_tea = torch.split(logits_teacher,
                                                                 [2 * self.args.batch_size, 2 * self.args.batch_size],
                                                                 dim=0)

                logits_ori_tea, logits_aug_tea = logits_ori_tea.detach(), logits_aug_tea.detach()
                assert logits_ori.size(0) == logits_aug.size(0)

                # classification loss for original data
                loss_cls = F.cross_entropy(logits_ori, labels)

                # classification loss for augmented data
                loss_aug = F.cross_entropy(logits_aug, labels)

                # calculate probability
                p_ori, p_aug = F.softmax(logits_ori / 10.0, dim=1), F.softmax(logits_aug / 10.0, dim=1)
                p_ori_tea, p_aug_tea = F.softmax(logits_ori_tea / 10.0, dim=1), F.softmax(logits_aug_tea / 10.0, dim=1)

                # use KLD for consistency loss
                loss_ori_tea = F.kl_div(p_aug.log(), p_ori_tea, reduction='batchmean')
                loss_aug_tea = F.kl_div(p_ori.log(), p_aug_tea, reduction='batchmean')

                const_weight = get_current_consistency_weight(epoch=current_epoch,
                                                              weight=self.weight,
                                                              rampup_length=self.args.rampup_length,
                                                              rampup_type='sigmoid')

                # calculate total loss
                total_loss += 0.5 * loss_cls + 0.5 * loss_aug + \
                              const_weight * loss_ori_tea + const_weight * loss_aug_tea

        total_loss /= len(minibatches)
        opt.zero_grad()
        self.scaler.scale(total_loss).backward()  # 为了梯度放大
        self.scaler.step(
            opt)  # 首先把梯度值unscale回来，如果梯度值不是inf或NaN,则调用optimizer.step()来更新权重，否则，忽略step调用，从而保证权重不更新。　　 scaler.step(optimizer)
        self.scaler.update()  # 准备着，看是否要增大scaler

        self.teacher_network.update(self.network)
        if sch:
            sch.step()
        self.update_count += 1
        return {'class': loss_cls.item(), 'loss_aug': loss_aug.item(), 'loss_ori_tea': loss_ori_tea.item(),
                'loss_aug_tea': loss_aug_tea.item(),
                'total': total_loss.item()}
