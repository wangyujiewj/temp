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
from hier_modelv3 import MatchNet_0, MatchNet_1, MatchNet_2, HMNet

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')

def train(args, train_loader, test_loader):
    # first stage
    tem = args.temperature
    model_0 = MatchNet_0(args).cuda()
    opt_0 = optim.Adam(filter(lambda p: p.requires_grad, model_0.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler_0 = ReduceLROnPlateau(opt_0, patience=8, verbose=True, mode='max')
    net = HMNet(args, model_0).cuda()
    for epoch in range(int(args.epochs * 0.1)):
        info_train = net._train_one_epoch(epoch=epoch, train_loader=train_loader, opt=opt_0, temp=tem)
        info_test = net._test_one_epoch(epoch=epoch, test_loader=test_loader, temp=tem)
        tem = tem * 0.98
        info_test_best = None
        r2_score = info_test['r_ab_r2_score']
        if info_test_best is None or info_test_best['r_ab_r2_score'] < info_test['r_ab_r2_score']:
            info_test_best = info_test
            info_test_best['stage'] = 'best_test'
            path = 'checkpoints/%s/models/model.best.t7' % args.exp_name
            net.save(path)
        net.logger.write(info_test_best)
        # 调整学习率，相当于step
        scheduler_0.step(r2_score)
        net.save('checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()
    # second stage
    model_1 = MatchNet_1(args).cuda()
    tem = args.temperature
    opt_1 = optim.Adam(filter(lambda p: p.requires_grad, model_1.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler_1 = ReduceLROnPlateau(opt_1, patience=8, verbose=True, mode='max')
    net = HMNet(args, model_1).cuda()
    info_test_best = None
    for epoch in range(int(args.epochs * 0.1), int(args.epochs * 0.2)):
        info_train = net._train_one_epoch(epoch=epoch, train_loader=train_loader, opt=opt_1, temp=tem)
        info_test = net._test_one_epoch(epoch=epoch, test_loader=test_loader, temp=tem)
        tem = tem * 0.98
        r2_score = info_test['r_ab_r2_score']
        if info_test_best is None or info_test_best['r_ab_r2_score'] < info_test['r_ab_r2_score']:
            info_test_best = info_test
            info_test_best['stage'] = 'best_test'
            path = 'checkpoints/%s/models/model.best.t7' % args.exp_name
            net.save(path)
        net.logger.write(info_test_best)
        # 调整学习率，相当于step
        scheduler_1.step(r2_score)
        net.save('checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()
    # third stage
    model_2 = MatchNet_2(args).cuda()
    tem = args.temperature
    opt_2 = optim.Adam(filter(lambda p: p.requires_grad, model_2.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler_2 = ReduceLROnPlateau(opt_2, patience=8, verbose=True, mode='max')
    info_test_best = None
    net = HMNet(args, model_2).cuda()
    for epoch in range(int(args.epochs * 0.2), args.epochs):
        info_train = net._train_one_epoch(epoch=epoch, train_loader=train_loader, opt=opt_2, temp=tem)
        info_test = net._test_one_epoch(epoch=epoch, test_loader=test_loader, temp=tem)
        tem = tem * 0.98
        r2_score = info_test['r_ab_r2_score']
        if info_test_best is None or info_test_best['r_ab_r2_score'] < info_test['r_ab_r2_score']:
            info_test_best = info_test
            info_test_best['stage'] = 'best_test'
            path = 'checkpoints/%s/models/model.best.t7' % args.exp_name
            net.save(path)
        net.logger.write(info_test_best)
        # 调整学习率，相当于step
        scheduler_2.step(r2_score)
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
    else:
        raise Exception("not implemented")
    if args.model == 'hmnet':
        if args.eval:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            model_2 = MatchNet_2(args)
            net = HMNet(args, model_2)
            eval_model(net, test_loader)
        else:
            train(args, train_loader, test_loader)
    else:
        raise Exception('Not implemented')

    print('FINISH')

if __name__ == '__main__':
    main()
