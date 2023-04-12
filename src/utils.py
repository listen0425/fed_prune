#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    下载数据用的
    return：训练集，测试集，用户字典（键是第几个用户，值是每个图片的序号；）
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        # 初始化

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    传入的是users个的各自模型。
    """

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():# 一共几层
        for i in range(1, len(w)): # len(w)为多少个user
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))# 归一
    return w_avg

def prune_avg_weights(w,all_prlist):
    prune_tu={}
    print(all_prlist)
    for key1 in all_prlist.keys():
        contents=all_prlist[key1]
        print(key1,'contents')
        print(contents)
        for key2 in contents.keys():
            content=contents[key2]
            if key2 not in prune_tu.keys():
                prune_tu[key2]=set(content)
            else:
                mid=set(content)
                prune_tu[key2]=prune_tu[key2].union(mid)

    w_avg = copy.deepcopy(w[0])
    for key in w[0].keys():
        w_avg[key].zero_()
    for i in range(len(w)):  # len(w)为多少个user
        check=0 # 记录总共多少层
        for key in w_avg.keys():  # 一共几层
            if key in prune_tu.keys():
                if check not in prune_tu[key]:
                    w_avg[key] += w[i][key]
                else:
                    continue
            else:
                w_avg[key] += w[i][key]
            check+=1
    for key in w_avg.keys():  # 一共几层
        w_avg[key] = torch.div(w_avg[key], len(w))  # 归一

    return prune_tu, w_avg
    # prune_tu是一个字典，用来存每一个剪枝的层被剪掉不更新的通道数；
    # w_avg是对平均未剪枝层的更新参数，剪枝层全部置0，后续重新分发




def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
