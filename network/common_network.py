# coding=utf-8
import torch.nn as nn
import math
import torch
import torch.nn.utils.weight_norm as weightNorm


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        # self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

# class head(nn.Module):
#     def __init__(self,feature_dim):
#         super(head, self).__init__()
#         self.head = nn.Sequential(
#                 nn.Linear(feature_dim, feature_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(feature_dim, feature_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(feature_dim, 128)
#             )
#     def forward(self, x):
#         return nn.functional.normalize(self.head(x),dim=1)

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(
                nn.Linear(bottleneck_dim, class_num), name="weight")
            # self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            # self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        # self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        # self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x


class MCdropClassifier(nn.Module):
    def __init__(self, in_features, num_classes, bottleneck_dim=512, dropout_rate=0.5, dropout_type='Bernoulli'):
        super(MCdropClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type

        self.bottleneck_drop = self._make_dropout(dropout_rate, dropout_type)

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim),
            nn.ReLU(),
            self.bottleneck_drop
        )

        self.prediction_layer = nn.Linear(bottleneck_dim, num_classes)

    def _make_dropout(self, dropout_rate, dropout_type):
        if dropout_type == 'Bernoulli':
            return nn.Dropout(dropout_rate)
        elif dropout_type == 'Gaussian':
            return GaussianDropout(dropout_rate)
        else:
            raise ValueError(f'Dropout type not found')

    def activate_dropout(self):
        self.bottleneck_drop.train()

    def forward(self, x):
        hidden = self.bottleneck_layer(x)
        pred = self.prediction_layer(hidden)
        return pred

class GaussianDropout(nn.Module):
    def __init__(self, drop_rate):
        super(GaussianDropout, self).__init__()
        self.drop_rate = drop_rate
        self.mean = 1.0
        self.std = math.sqrt(drop_rate / (1.0 - drop_rate))

    def forward(self, x):
        if self.training:
            gaussian_noise = torch.randn_like(x, requires_grad=False).to(x.device) * self.std + self.mean
            return x * gaussian_noise
        else:
            return x


def encoder(args):
    if args.net == 'resnet50':
        n_outputs = 2048
    elif args.net == 'resnet18':
        n_outputs = 512
    else:
        n_outputs = 1024
    if args.dataset == "office-home":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        encoder = nn.Sequential(
            nn.Linear(n_outputs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(512, 512),
        )
    elif args.dataset == "PACS":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        encoder = nn.Sequential(
            nn.Linear(n_outputs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(512, 256),
        )

    elif args.dataset == "terra_incognita":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        encoder = nn.Sequential(
            nn.Linear(n_outputs,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(512, 512),
        )
    else:
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        encoder = nn.Sequential(
            nn.Linear(n_outputs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(128, 128),
        )

    return encoder, scale_weights, pcl_weights


def fea_proj(args):
    if args.dataset == "office-home":
        dropout = nn.Dropout(0.25)
        fea_proj = nn.Sequential(
            nn.Linear(512,
                      512),
            dropout,
            nn.Linear(512,
                      512),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(512,
                              512)
        )
    elif args.dataset == "PACS":
        dropout = nn.Dropout(0.25)
        fea_proj = nn.Sequential(
            nn.Linear(256,
                      256),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(256,
                              256)
        )

    elif  args.dataset  == "terra_incognita":
        dropout = nn.Dropout(0.25)
        fea_proj = nn.Sequential(
            nn.Linear(512,
                      512),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(512,
                              512)
        )
    else:
        dropout = nn.Dropout(0.25)
        fea_proj = nn.Sequential(
            nn.Linear(128,
                      128),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(128,
                              128)
        )

    return fea_proj, fc_proj
