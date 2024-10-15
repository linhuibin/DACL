import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sagm import SAGM
from utils.scheduler import LinearScheduler
from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from alg.opt import *


class SAGM_DG(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    # def __init__(self, input_shape, num_classes, num_domains, hparams):
    #     assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
    #     super().__init__(input_shape, num_classes, num_domains, hparams)
    def __init__(self, args):
        super(SAGM_DG, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer1(
            self.network.parameters(),
            args
        )
        self.lr_scheduler = LinearScheduler(T_max=5000, max_value=args.lr,
                                            min_value=args.lr, optimizer=self.optimizer)

        self.rho_scheduler = LinearScheduler(T_max=5000, max_value=0.05,
                                             min_value=0.05)

        self.SAGM_optimizer = SAGM(params=self.network.parameters(), base_optimizer=self.optimizer, model=self.network,
                                   alpha=5e-4, rho_scheduler=self.rho_scheduler, adaptive=False)

    def update(self, minibatches):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        self.SAGM_optimizer.set_closure(loss_fn, all_x, all_y)
        predictions, loss = self.SAGM_optimizer.step()
        self.lr_scheduler.step()
        self.SAGM_optimizer.update_rho_t()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)
