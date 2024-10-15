# coding=utf-8
import os
import time
import numpy as np
from arg_utils import get_args

import torch.nn

from utils.compute_std import compute_std
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, load_checkpoint, print_args, train_valid_target_eval_names, \
    alg_loss_dict
from datautil.getdataloader import get_img_dataloader
from torch.utils.tensorboard import SummaryWriter



def main():
    args = get_args()
    set_random_seed(args.seed)

    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    # algorithm = torch.nn.DataParallel(algorithm)
    algorithm = torch.compile(algorithm)
    # args = load_checkpoint('model.pkl', algorithm, args)
    algorithm.train()
    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)
    # writer = SummaryWriter(args.output)
    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)
    if 'DIFEX' in args.algorithm:
        ms = time.time()
        n_steps = args.max_epoch * args.steps_per_epoch
        print('start training fft teacher net')
        opt1 = get_optimizer(algorithm.teaNet, args, isteacher=True)
        sch1 = get_scheduler(opt1, args)
        algorithm.teanettrain(train_loaders, n_steps, opt1, sch1)
        print('complet time:%.4f' % (time.time() - ms))

    acc_record = {}
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0
    print('===========start training===========')
    sss = time.time()
    for epoch in range(args.max_epoch):
        for iter_num in range(args.steps_per_epoch):
            minibatches_device = [(data)
                                  for data in next(train_minibatches_iterator)]
            # Reset optim, because it doesn't like the sharp jump in gradient magnitudes that happens at this step.
            if (args.algorithm == 'VREx' or args.algorithm == 'IRM') and algorithm.update_count == args.anneal_iters:
                opt = get_optimizer(algorithm, args)
                sch = get_scheduler(opt, args)
            step_vals = algorithm.update(minibatches_device, opt, sch)

        if (epoch in [int(args.max_epoch * 0.7), int(args.max_epoch * 0.9)]) and (not args.schuse):
            print('manually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr'] * 0.1

        # Visulization
        # for item in loss_list:
        #     writer.add_scalar(f'loss/{item}_loss', step_vals[item], epoch)  # visualize loss
        # for item in acc_type_list:
        #     acc_record[item] = np.mean(np.array([modelopera.accuracy(
        #         algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
        #     writer.add_scalar(f'acc/{item}_acc', acc_record[item], epoch)  # visualize acc

        if (epoch == (args.max_epoch - 1) or (epoch % args.checkpoint_freq == 0)):
            print('===========epoch %d===========' % (epoch))
            s = ''
            for item in loss_list:
                s += (item + '_loss:%.4f,' % step_vals[item])  # print all loss respectively
            print(s[:-1])
            s = ''
            for item in acc_type_list:
                if item != 'train':
                    acc_record[item] = np.mean(np.array([modelopera.accuracy(
                        algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
                    s += (item + '_acc:%.4f,' % acc_record[item])
                # print all accuracy respectively, train, val, target domain respectively
            print(s[:-1])

            if acc_record['valid'] > best_valid_acc:
                best_valid_acc = acc_record['valid']
                target_acc = acc_record['target']
                save_checkpoint('model_best.pkl', algorithm, args)
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_epoch{epoch}.pkl', algorithm, args)
            print('total cost time: %.4f' % (time.time() - sss))
            # algorithm_dict = algorithm.state_dict()
            # print(sch.get_last_lr()[0])
    save_checkpoint('model.pkl', algorithm, args)
    print('valid acc: %.4f' % best_valid_acc)
    print('DG result: %.4f' % target_acc)

    # writer.close()
    with open(os.path.join(args.output, 'done.txt'), mode="a") as f:
        f.write('total cost time:%s\n' % (str(time.time() - sss)))
        f.write('valid acc:%.4f\n' % (best_valid_acc))
        f.write('target acc seed%d:%.4f\n\n' % (args.seed, target_acc))
    f.close()
    if args.compute_std and args.seed == 3:
        compute_std(args.output)


if __name__ == '__main__':
    main()
