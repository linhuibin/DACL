# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM
import torch.autograd as autograd


class ANDMask(ERM):
    def __init__(self, args):
        super(ANDMask, self).__init__(args)

        self.tau = args.tau    # sensitive for tau

    def update(self, minibatches, opt, sch):

        total_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, data in enumerate(minibatches):  # 最后还要再除以minibatches的长度
            x, y = data[0].cuda().float(), data[1].cuda().long()
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]  # 对应当前env的预测值
            all_logits_idx += x.shape[0]

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grads = autograd.grad(
                env_loss, self.network.parameters(), retain_graph=True)  # 每次循环都要计算一次梯度，所以要保存图
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)  # 更改 grads，param_gradients 也会变化
        # param_gradients contains gradients for all environment parameters e.g: source domain=3 : list [ list1[env1][env2][env3], list2[env1][env2][env3],  ]

        mean_loss = total_loss / len(minibatches)

        opt.zero_grad()
        self.mask_grads(param_gradients, self.network.parameters())
        opt.step()
        if sch:
            sch.step()

        return {'total': mean_loss.item()}

    def mask_grads(self, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())   # mask percentage
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))  # The total value remains relatively unchanged
        return 0
