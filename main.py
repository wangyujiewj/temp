#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import gc
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import ModelNet40
from arguments import hmnet_arguments
import numpy as np
from torch.utils.data import DataLoader
from model_ransac import HMNet
from rpm_data import get_train_datasets
import torch.nn as nn

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')

def adjust_learning_rate(opt, epoch, wait_epoch):
    # 在第10个epoch调大学习率
    if epoch == 9:
        opt.lr = opt.lr * 10
    if wait_epoch > 8:
        opt.lr = opt.lr * 0.1
        wait_epoch = 0
    # 更新参数的学习率
    for param_group in opt.param_groups:
        param_group['lr'] = opt.lr
    return wait_epoch

def train(args, net, train_loader, test_loader):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = ReduceLROnPlateau(opt, patience=8, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    info_test_best = None
    wait_epoch = 0
    tem = args.temperature
    for epoch in range(args.epochs):
        info_train = net._train_one_epoch(epoch=epoch, train_loader=train_loader, opt=opt, temp=tem)
        info_test = net._test_one_epoch(epoch=epoch, test_loader=test_loader, temp=tem)
        tem = tem * 0.98
        # val_loss = info_test['loss']
        if info_test_best is None or info_test_best['r_ab_r2_score'] < info_test['r_ab_r2_score']:
            info_test_best = info_test
            info_test_best['stage'] = 'best_test'
            path = 'checkpoints/%s/models/model.best.t7' % args.exp_name
            net.save(path)
        else:
            wait_epoch = wait_epoch + 1
        net.logger.write(info_test_best)
        # 调整学习率，相当于step
        wait_epoch = adjust_learning_rate(opt, epoch, wait_epoch)
        # scheduler.step(val_loss)
        net.save('checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()

def eval_model(net, test_loader):
    tem = 0.135326
    epoch = 1
    net.eval()
    net._test_one_epoch(epoch=epoch, test_loader=test_loader, temp=tem)

def main():
    parser = hmnet_arguments()
    args = parser.parse_args()
    # 保证实验的可重复性
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    _init_(args)
    if args.dataset == 'ours':
        train_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                             num_subsampled_points=args.n_subsampled_points,
                                             partition='train', gaussian_noise=args.gaussian_noise,
                                             rot_factor=args.rot_factor, overlap=args.overlap),
                                  batch_size=args.train_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        test_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                            num_subsampled_points=args.n_subsampled_points,
                                            partition='test', gaussian_noise=args.gaussian_noise,
                                            rot_factor=args.rot_factor, overlap=args.overlap),
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    elif args.dataset == 'rpmnet':
        train_set, val_set = get_train_datasets(args)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.train_batch_size, shuffle=True,
                                                   num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=args.val_batch_size, shuffle=False,
                                                 num_workers=args.num_workers)
    else:
        raise Exception("not implemented")
    if args.model == 'hmnet':
        net = HMNet(args).cuda()
        if args.eval:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            eval_model(net, test_loader)
        else:
            train(args, net, train_loader, test_loader)
    else:
        raise Exception('Not implemented')

    print('FINISH')

if __name__ == '__main__':
    main()
