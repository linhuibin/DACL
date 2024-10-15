# coding=utf-8
import torch
from network import img_network


def get_fea(args):
    if args.dataset == 'digits_dg' or args.dataset == 'dg5':
        net = img_network.cnn_digitsingle()
    elif args.net.startswith('res'):
        if args.algorithm == 'DNA' or args.algorithm == 'PCL' or args.algorithm == 'SAGM_DG':
            net = img_network.ResNet_DAN(args.net)
        else:
            net = img_network.ResBase(args.net)
    else:
        net = img_network.VGGBase(args.net)
    return net


def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total


def accuracy_pcl(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            _, p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total