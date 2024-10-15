# coding=utf-8
import random
import numpy as np
import sys
import os
import torchvision
import PIL
import argparse
import torch
from torch.nn.modules.batchnorm import _BatchNorm


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(filename, alg, args):
    save_dict = {
        "args": vars(args),
        "model_dict": alg.state_dict()
    }
    torch.save(save_dict, os.path.join(args.output, filename))


def load_checkpoint(filename, alg, args):
    checkpoint = torch.load(os.path.join(args.output, filename),
                            map_location=torch.device('cuda', 0))
    model = checkpoint['model_dict']
    # keys_list = list(model.keys())
    # for key in keys_list:
    #     if 'orig_mod.' in key:
    #         deal_key = key.replace('_orig_mod.', '')
    #         model[deal_key] = model[key]
    #         del model[key]
    args = argparse.Namespace(**checkpoint['args'])
    alg.load_state_dict(model)
    return args


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict


def train_valid_target_eval_names_digits(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
            t += 1
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict


def train_valid_target_eval_single(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}

    eval_name_dict['train'].append(0)
    if args.test_envs[0] > args.train_envs[0]:
        eval_name_dict['valid'].append(1)
        eval_name_dict['target'].append(2)
    else:
        eval_name_dict['target'].append(1)
        eval_name_dict['valid'].append(2)
    return eval_name_dict


def train_valid_target_eval_digits_single(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}

    eval_name_dict['train'].append(0)
    if args.test_envs[0] > args.train_envs[0]:
        eval_name_dict['valid'].append(1)
        eval_name_dict['target'].append(2)
        eval_name_dict['target'].append(3)
    else:
        eval_name_dict['target'].append(1)
        eval_name_dict['target'].append(2)
        eval_name_dict['valid'].append(3)
    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {'ANDMask': ['total'],
                 'CORAL': ['class', 'coral', 'total'],
                 'DANN': ['class', 'dis', 'total'],
                 'ERM': ['class'],
                 'Mixup': ['class'],
                 'Mixup1': ['class'],
                 'MLDG': ['total'],
                 'MMD': ['class', 'mmd', 'total'],
                 'GroupDRO': ['group'],
                 'RSC': ['class'],
                 'VREx': ['loss', 'nll', 'penalty'],
                 'IRM': ['loss', 'nll', 'penalty'],
                 'MTL': ['loss'],
                 'DIFEX': ['class', 'dist', 'exp', 'align', 'total'],
                 'FACT': ['class', 'loss_aug', 'loss_ori_tea', 'loss_aug_tea', 'total'],
                 'DNA': ['total', 'loss_c', 'loss_v'],
                 'DACL': ['total', 'class', 'consistency', 'contrast_loss'],
                 'PCL': ['total', 'loss_cls', 'loss_pcl'],
                 'SAGM_DG': ['loss'],
                 }

    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-caltech':
        domains = ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    elif dataset == 'digits_dg':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn']
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'VLCS':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset == 'terra_incognita':
        domains = ['location_100', 'location_38', 'location_43', 'location_46']
    elif dataset == 'domainnet':
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'office': ['amazon', 'dslr', 'webcam'],
        'office-caltech': ['amazon', 'dslr', 'webcam', 'caltech'],
        'office-home': ['Art', 'Clipart', 'Product', 'RealWorld'],
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'digits_dg': ['mnist', 'mnist_m', 'svhn', 'syn'],
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
        'terra_incognita': ['location_100', 'location_38', 'location_43', 'location_46'],
        'domainnet': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    }
    if dataset == 'digits_dg' or dataset == 'dg5':
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    else:
        args.input_shape = (3, 224, 224)
        if args.dataset == 'office-home':
            args.num_classes = 65
        elif args.dataset == 'office':
            args.num_classes = 31
        elif args.dataset == 'PACS':
            args.num_classes = 7
        elif args.dataset == 'VLCS':
            args.num_classes = 5
        elif args.dataset == 'terra_incognita':
            args.num_classes = 10
        elif args.dataset == 'domainnet':
            args.num_classes = 345
    return args


def get_current_consistency_weight(epoch, weight, rampup_length, rampup_type='step'):
    if rampup_type == 'step':
        rampup_func = step_rampup
    elif rampup_type == 'linear':
        rampup_func = linear_rampup
    elif rampup_type == 'sigmoid':
        rampup_func = sigmoid_rampup
    else:
        raise ValueError("Rampup schedule not implemented")

    return weight * rampup_func(epoch, rampup_length)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def step_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0.0


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def PJS_loss(prob, label):
    row_index = torch.arange(0, prob.size(0))
    prob_y = prob[row_index, label]
    loss = (torch.log(2 / (1 + prob_y)) + prob_y * torch.log(2 * prob_y / (1 + prob_y))).mean()
    return loss


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
