import argparse
import os
import sys
from utils.util import Tee, img_param_init, print_environ


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="DACL")
    parser.add_argument('--alpha', type=float,
                        default=0.5, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--beta', type=float,
                        default=0.01, help='DIFEX beta')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam hyper-param')
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=1, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='PACS')
    parser.add_argument('--data_dir', type=str, default='/mlspace/datasets/PACS/', help='data dir')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--disttype', type=str, default='2-norm',
                        choices=['1-norm', '2-norm', 'cos', 'norm-2-norm', 'norm-1-norm'])
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='1', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lam', type=float,
                        default=0.1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd cosine scheduler')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=120, help="max iterations")
    parser.add_argument('--mixupalpha', type=float,
                        default=0.1, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=0.5, help='MMD, CORAL hyper-param')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet18',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase")
    parser.add_argument('--N_WORKERS', type=int, default=0)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1 / 3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1 / 3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg", \
                        choices=["img_dg", 'img_dg_single'], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=0.82, help="AndMask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[0], help='target domains')
    parser.add_argument('--output', type=str,
                        default="/mlspace/linhb/DeepDG/scripts/PACS_resnet18/env0/DACL",
                        help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--compute_std', type=bool, default=True)
    parser.add_argument('--weight', type=float, default=2.0, help="FACT weight")
    parser.add_argument('--rampup_length', type=int, default=5, help="FACT rampup_length")
    parser.add_argument('--mtl_ema', type=float, default=0.99)
    parser.add_argument('--ema_decay', type=float, default=0.9995)
    parser.add_argument('--optimizer', type=str, default='SGD', help='type of optimizer')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention head')
    parser.add_argument('--T', type=float, default=10, help='temperature of prediction')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature of contrastive')
    parser.add_argument('--metric', type=str, default="euclidean", help='temperature of contrastive')
    parser.add_argument('--uniform', type=float, default=1.0, help='range of uniform')
    parser.add_argument('--pk', type=int, default=20, help='number of prototypes')
    parser.add_argument('--qratio', type=int, default=1, help='number of prototypes')
    parser.add_argument('--ratio', type=float, default=0.25, help='furrier ratio')
    parser.add_argument('--amp', '-a', action='store_true', help='if specified, turn amp on')
    parser.add_argument('--model_type', type=str, default='clip', help='the type of model (CLIP or ResNet)')

    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = args.data_file + args.data_dir
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args
