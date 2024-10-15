# Distribution Alignment Using Prototypes Contrastive Loss Ours
import torch
import torch.nn as nn
import torch.nn.functional as F
from alg.modelopera import get_fea
import numpy as np
from network import common_network
from alg.algs.base import Algorithm
from network.atten_network import AttenHead
from network.ema_dapc import ModelEMA
from utils.util import get_current_consistency_weight
from network.loss import ProtoLoss
from sklearn.cluster import KMeans
import random
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

class DACL(Algorithm):
    def __init__(self, args):
        super(DACL, self).__init__(args)
        self.steps_per_epoch = args.steps_per_epoch
        self.weight = args.weight
        self.T = args.T
        self.metric = args.metric
        self.update_count = 0
        self.ratio = args.ratio
        self.featurizer = get_fea(args)
        in_features = self.featurizer.in_features
        # self.featurizer = nn.DataParallel(self.featurizer)

        self.num_classes = args.num_classes
        self.td_num = args.domain_num - len(args.test_envs)

        self.atten_head = AttenHead(in_features, args.num_heads)
        # self.atten_head = nn.DataParallel(self.atten_head)

        self.classifier = common_network.feat_classifier(
            args.num_classes, in_features, args.classifier)
        # self.classifier = nn.DataParallel(self.classifier)

        self.teacher_network = ModelEMA(
            self.featurizer, self.classifier, args)
        self.criterion = ProtoLoss(args.temperature, args.num_classes, self.td_num)
        self.qratio = args.qratio
        self.qs = args.batch_size * self.td_num * self.qratio * 100
        self.uniform = args.uniform
        self.pk = args.pk
        self.register_buffer("all_z", torch.randn(self.qs, in_features))
        self.register_buffer("all_yl", torch.zeros(self.qs, dtype=torch.long))
        self.register_buffer("all_dl", torch.zeros(self.qs, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.amp = args.amp
        self.scaler = GradScaler(enabled=args.amp)


    def update(self, minibatches, opt, sch):
        ptr = int(self.queue_ptr)
        x = torch.cat(
            [data[0].cuda().float() for data in minibatches])  # all_x refers to the data of all domains in a minibatch
        yl = torch.cat([data[1].cuda().long() for data in minibatches])
        dl = torch.cat([data[2].cuda().long() for data in minibatches])

        x_size = x.size(0)

        self.all_yl[ptr:(ptr + x_size)] = yl
        self.all_dl[ptr:(ptr + x_size)] = dl

        fft_x = torch.fft.fftn(x, dim=(-2, -1))
        amp, pha = torch.abs(fft_x), torch.angle(fft_x)
        amp_avg = []
        for di in torch.unique(dl, sorted=True):
            amp_avg_di = torch.mean(amp[di == dl], dim=0, keepdim=True)
            amp_avg.append(amp_avg_di)
        amp_avg = torch.cat(amp_avg).cuda()

        l = np.random.uniform(0, self.uniform)

        if random.random() <= 0.5:
            amp_avgi = amp_avg[
                   (torch.tensor(np.random.randint(0, self.td_num, size=x_size)).cuda())]
        else:
            amp_avgi = amp_avg[
                 (dl + torch.tensor(np.random.randint(1, self.td_num, size=x_size)).cuda()) % self.td_num]

        amp_mix = (1 - l) * amp + l * amp_avgi
        _, _, rows, cols = x.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        f_amp_mix = torch.fft.fftshift(amp_mix, dim=(-2, -1))
        mask = torch.zeros([rows, cols]).cuda()
        hrows = int(rows * self.ratio)
        mask[crow - hrows:crow + hrows, ccol - hrows:ccol + hrows] = 1  # 低通滤波器
        f_amp_mix = f_amp_mix * mask
        amp = torch.fft.ifftshift(f_amp_mix, dim=(-2, -1))

        fft_src_ = amp * (torch.exp(1j * pha))
        x_aug = torch.real(torch.fft.ifftn(fft_src_, dim=(-2, -1)))

        inputs = torch.cat((x, x_aug), dim=0)

        with autocast(enabled=self.amp):
            feature = self.featurizer(inputs)
            prediction = self.classifier(feature)
            ori_fea, aug_fea = torch.split(feature, [x_size, x_size], dim=0)
            ori_logits, aug_logits = torch.split(prediction, [x_size, x_size], dim=0)

            t_ori_fea = self.teacher_network.featurizer(x)
            t_ori_logits = self.teacher_network.classifier(t_ori_fea)

            self.all_z[ptr:ptr + x_size, :] = t_ori_fea
            ptr = (ptr + x.size(0)) % self.qs
            self.queue_ptr[0] = ptr

            class_loss = (F.cross_entropy(ori_logits, yl) + F.cross_entropy(aug_logits, yl))

            if self.update_count < self.qratio * self.steps_per_epoch:
                total_loss = class_loss
            else:
                const_weight = get_current_consistency_weight(epoch=(self.update_count / self.steps_per_epoch) - self.qratio,
                                                              weight=self.weight,
                                                              rampup_length=5,
                                                              rampup_type='sigmoid')
                p_ori_tea = torch.softmax(t_ori_logits / self.T, dim=-1)
                p_ori = torch.softmax(ori_logits / self.T, dim=-1)
                con_loss1 = F.kl_div(p_ori.log(), p_ori_tea, reduction='batchmean')
                # 注意力机制
                fxg, _ = self.atten_head(ori_fea, self.fp)
                logits_g = self.classifier(fxg)

                p_g = F.softmax(logits_g / self.T, dim=-1)
                p_aug = torch.softmax(aug_logits / self.T, dim=-1)


                a1 = 0.5 * (p_g + p_aug)
                loss1 = -torch.mean(torch.sum(torch.log(a1) * a1, dim=1))
                loss2 = -torch.mean(torch.sum(torch.log(p_g) * p_g, dim=1))
                loss3 = -torch.mean(torch.sum(torch.log(p_aug) * p_aug, dim=1))
                con_loss = loss1 - 0.5 * (loss2 + loss3)

                contrast_loss = self.criterion(feature, self.fp, torch.cat([yl, yl]), self.yp, torch.cat([dl, dl]),
                                               self.dp, self.score)

                total_loss = class_loss + const_weight * (con_loss1 + con_loss) + 0.5 * (contrast_loss)

        opt.zero_grad()
        self.scaler.scale(total_loss).backward()  # 为了梯度放大
        self.scaler.step(opt)  # 首先把梯度值unscale回来，如果梯度值不是inf或NaN,则调用optimizer.step()来更新权重，否则，忽略step调用，从而保证权重不更新。　　 scaler.step(optimizer)
        self.scaler.update()  # 准备着，看是否要增大scaler

        # opt.step()
        self.teacher_network.update(nn.Sequential(
            self.featurizer, self.classifier))
        if sch:
            sch.step()

        self.update_count += 1
        if self.update_count % (self.steps_per_epoch) == 0:
            self.extract_fp(self.pk)

        return {'total': total_loss.item(), 'class': class_loss.item(), 'consistency': 0,
                'contrast_loss': 0}

    def extract_fp(self, pk):
        # pk prototype 数量
        fp, dp, yp, s, mean_d = [], [], [], [], []  # fp: feature of prototypes  yp: label of prototypes lp: 判断是否为标记或未标记
        for di in torch.unique(self.all_dl, sorted=True):
            zd = self.all_z[self.all_dl == di]
            mean_zd = torch.mean(zd, dim=0, keepdim=True)
            cent_z = zd - mean_zd
            cova_z = (cent_z.t() @ cent_z) / (len(zd) - 1)
            mean_d.append(mean_zd)
            s.append(cova_z)
            for yi in torch.unique(self.all_yl, sorted=True):
                # prototypes extracted from labeled data
                fx = self.all_z[(self.all_yl == yi) * (self.all_dl == di)]
                if (fx.numel()):  # fx no empty
                    fpi = self.extract_fp_per_class(fx, pk)  # idea 原型标记10个，
                    fp.append(fpi)
                    pkl = len(fpi)
                    yp.append(torch.full((pkl,), yi, dtype=torch.long))
                    dp.append(torch.full((pkl,), di, dtype=torch.long))


        self.dp = torch.cat(dp).cuda()
        self.fp = torch.cat(fp).cuda()
        self.yp = torch.cat(yp).cuda()
        self.mean_d = torch.cat(mean_d).cuda()
        self.score = torch.ones([self.td_num, self.td_num]).cuda()
        for di in torch.unique(self.all_dl, sorted=True):
            for dj in torch.unique(self.all_dl, sorted=True):
                sim = (s[di] * s[dj]).sum() / (torch.norm(s[di]) * torch.norm(s[dj]))
                dis = (1 - sim)
                self.score[di][dj] = dis

    def extract_fp_per_class(self, fx, n, record_mean=True):  # record_mean=True 取该类均值
        if n == 1:
            fp = torch.mean(fx, dim=0, keepdim=True)
        elif record_mean:
            n = n - 1
            fm = torch.mean(fx, dim=0, keepdim=True)
            if n >= len(fx):
                fp = fx
            else:
                fp = self.kmeans(fx, n, self.metric)
            fp = torch.cat([fm, fp], dim=0)
        else:
            if n >= len(fx):
                fp = fx
            else:
                fp = self.kmeans(fx, n, self.metric)
        return fp

    @staticmethod
    def kmeans(fx, n, metric='euclidean'):
        device = fx.device
        if metric == 'cosine':
            fn = fx / torch.clamp(torch.norm(fx, dim=1, keepdim=True), min=1e-20)
        elif metric == 'euclidean':
            fn = fx
        else:
            raise KeyError
        fn = fn.detach().cpu().numpy()
        fx = fx.detach().cpu().numpy()

        labels = KMeans(n_clusters=n, n_init='auto').fit_predict(fn)
        fp = np.stack([np.mean(fx[labels == li], axis=0) for li in np.unique(labels)])
        fp = torch.FloatTensor(fp).to(device)

        return fp

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def predict_val(self, x):
        ori_fea = self.featurizer(x)
        fxg, _ = self.atten_head(ori_fea, self.fp)
        return self.classifier(fxg)
