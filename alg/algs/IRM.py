import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM
import torch.autograd as autograd

class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, args):
        super(IRM, self).__init__(args)
        self.register_buffer('update_count', torch.tensor([0]))
        self.args = args

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[0::2] * scale, y[0::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, opt, sch):
        if self.update_count >= self.args.anneal_iters:
            penalty_weight = self.args.lam
        else:
            penalty_weight = 1.0

        nll = 0.
        penalty = 0.


        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, data in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + data[0].shape[0]]
            all_logits_idx += data[0].shape[0]
            nll += F.cross_entropy(logits, data[1].cuda().long())
            penalty += self._irm_penalty(logits, data[1].cuda().long())
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}
