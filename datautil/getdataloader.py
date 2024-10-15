# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader
import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader


def get_img_dataloader(args):
    if args.dataset in ('domainnet', 'terra_incognit', 'digits_dg', 'dg5', 'VLCS'):
        rate = 0.2
    else:
        rate = 0.1
    trdatalist, tedatalist = [], []
    weights = []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)


    image_train = imgutil.image_train
    image_test = imgutil.image_test

    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=image_test(args),
                                           test_envs=args.test_envs))
        else:
            tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=image_train(args), test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1 - rate,
                    random_state=args.seed)  # Extract data according to category
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l * rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            datasets = ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=image_train(args), indices=indextr,
                                    test_envs=args.test_envs)
            trdatalist.append(datasets)
            # weight = make_weights_for_balanced_classes(datasets.labels)
            # weights.append(weight[indextr])
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=image_test(args), indices=indexte,
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
