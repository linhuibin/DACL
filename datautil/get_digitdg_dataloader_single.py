# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader
import datautil.imgdata.util as imgutil
import os
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader


def get_img_dataloader(args):
    rate = 0.2  # validation set
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = 2

    image_train = imgutil.image_train
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           os.path.join(names[i], "train"), i, transform=imgutil.image_test(args.dataset),
                                           test_envs=args.test_envs))
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           os.path.join(names[i], "val"), i,
                                           transform=imgutil.image_test(args.dataset),
                                           test_envs=args.test_envs))
        elif i in args.train_envs:
            datasets = ImageDataset(args.dataset, args.task, args.data_dir,
                                       os.path.join(names[i], "train"), i, transform=image_train(args.dataset),
                                       test_envs=args.test_envs)         # train
            trdatalist.append(datasets)
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                              os.path.join(names[i], "val"), i, transform=imgutil.image_test(args.dataset),
                                              test_envs=args.test_envs))  # val

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=2 * args.N_WORKERS,
        pin_memory=True,
        drop_last=False,
        shuffle=False)
        for env in trdatalist + tedatalist]  # eval_loaders=[loader1, loader2]

    return train_loaders, eval_loaders
