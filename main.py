# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Project    : nnUNet
# File       : main_metatrain.py
# Time       ：2022/9/20 12:09
# Author     ：Yongheng Sun
"""
import pdb

from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, unpack_dataset
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
import numpy as np
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
import torch

from torch import nn
from nnunet.network_architecture.generic_UNet import Generic_UNet2
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
import torch

from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch.cuda.amp import GradScaler, autocast


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
# from eval import eval_dgnet
from tqdm import tqdm
import logging
# from metrics.focal_loss import FocalLoss
# from torch.utils.data import DataLoader, random_split, ConcatDataset
# import torch.nn.functional as F
# import utils
# from loaders.mms_dataloader_meta_split import get_meta_split_data_loaders
# import models
# import losses
import time


def cossim(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1).mean()

def get_args():
    usage_text = (
        "DGNet Pytorch Implementation"
        "Usage:  python train_meta.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-e','--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('-lr','--learning_rate', type=float, default=1e-3, help='The learning rate for model training')
    parser.add_argument('-lr2','--learning_rate2', type=float, default=1e-3, help='The learning rate for model training')
    parser.add_argument('--model_name', type=str, default='reg12w1-3w2-3lr-3', help= 'Path to save model checkpoints')
    parser.add_argument('--w1', type=float, default=0.1, help='The weight for l_inter')
    parser.add_argument('--w2', type=float, default=0.001, help='The weight for l_intra')
    parser.add_argument('--layer1', type=int, default=-1, help='The weight for l_inter')
    parser.add_argument('--layer2', type=int, default=87, help='The weight for l_intra')
    #hardware
    parser.add_argument('-g','--gpu', type=int, default=5, help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')


    return parser.parse_args()


def train_net(args):
    epochs = args.epochs
    layer1 = args.layer1
    layer2 = args.layer2
    model_name = args.model_name

    layers = []
    for i in torch.load('./checkpoints/ck.pth'):
        layers.append(i)

    #Model selection and initialization
    num_input_channels = 1
    base_num_features = 32
    num_classes = 4
    conv_per_stage = 2
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.ReLU
    net_nonlin_kwargs = {'inplace': True}
    net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    net_conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    network = Generic_UNet2(num_input_channels, base_num_features, num_classes,
                           5,
                           conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                           dropout_op_kwargs,
                           net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                           net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
    if torch.cuda.is_available():
        network.cuda(args.gpu)
    network.inference_apply_nonlin = softmax_helper




    basic_generator_patch_size = np.array([205, 205, 205])
    patch_size = np.array([128, 128, 128])
    batch_size = 2
    oversample_foreground_percent = 0.33

    # dataset_tr_501 = np.load('./data_npy/501_tr_fold4_3001.npy', allow_pickle=True).item()
    # dataset_val_501 = np.load('./data_npy/501_val_fold4_3001.npy', allow_pickle=True).item()
    dataset_tr_502 = np.load('./data_npy/502_tr_fold4_3001.npy', allow_pickle=True).item()
    dataset_val_502 = np.load('./data_npy/502_val_fold4_3001.npy', allow_pickle=True).item()
    # dataset_tr_503 = np.load('./data_npy/503_tr_fold4_3001.npy', allow_pickle=True).item()
    # dataset_val_503 = np.load('./data_npy/503_val_fold4_3001.npy', allow_pickle=True).item()
    dataset_tr_504 = np.load('./data_npy/504_tr_fold4_3001.npy', allow_pickle=True).item()
    dataset_val_504 = np.load('./data_npy/504_val_fold4_3001.npy', allow_pickle=True).item()
    dataset_tr_505 = np.load('./data_npy/505_tr_fold4_3001.npy', allow_pickle=True).item()
    dataset_val_505 = np.load('./data_npy/505_val_fold4_3001.npy', allow_pickle=True).item()

    # dl_tr_501 = DataLoader3D(dataset_tr_501, basic_generator_patch_size, patch_size, batch_size,
    #                      False, oversample_foreground_percent=oversample_foreground_percent,
    #                      pad_mode='constant', pad_sides=None, memmap_mode='r')
    # dl_val_501 = DataLoader3D(dataset_val_501, patch_size, patch_size, batch_size,
    #                       oversample_foreground_percent=oversample_foreground_percent,
    #                       pad_mode='constant', pad_sides=None, memmap_mode='r')
    dl_tr_502 = DataLoader3D(dataset_tr_502, basic_generator_patch_size, patch_size, batch_size,
                         False, oversample_foreground_percent=oversample_foreground_percent,
                         pad_mode='constant', pad_sides=None, memmap_mode='r')
    dl_val_502 = DataLoader3D(dataset_val_502, patch_size, patch_size, batch_size,
                          oversample_foreground_percent=oversample_foreground_percent,
                          pad_mode='constant', pad_sides=None, memmap_mode='r')
    # dl_tr_503 = DataLoader3D(dataset_tr_503, basic_generator_patch_size, patch_size, batch_size,
    #                      False, oversample_foreground_percent=oversample_foreground_percent,
    #                      pad_mode='constant', pad_sides=None, memmap_mode='r')
    # dl_val_503 = DataLoader3D(dataset_val_503, patch_size, patch_size, batch_size,
    #                       oversample_foreground_percent=oversample_foreground_percent,
    #                       pad_mode='constant', pad_sides=None, memmap_mode='r')
    dl_tr_504 = DataLoader3D(dataset_tr_504, basic_generator_patch_size, patch_size, batch_size,
                         False, oversample_foreground_percent=oversample_foreground_percent,
                         pad_mode='constant', pad_sides=None, memmap_mode='r')
    dl_val_504 = DataLoader3D(dataset_val_504, patch_size, patch_size, batch_size,
                          oversample_foreground_percent=oversample_foreground_percent,
                          pad_mode='constant', pad_sides=None, memmap_mode='r')
    dl_tr_505 = DataLoader3D(dataset_tr_505, basic_generator_patch_size, patch_size, batch_size,
                         False, oversample_foreground_percent=oversample_foreground_percent,
                         pad_mode='constant', pad_sides=None, memmap_mode='r')
    dl_val_505 = DataLoader3D(dataset_val_505, patch_size, patch_size, batch_size,
                          oversample_foreground_percent=oversample_foreground_percent,
                          pad_mode='constant', pad_sides=None, memmap_mode='r')

    # folder_with_preprocessed_data = '/home/omnisky/data1/sunyongheng/nnUNet/nnUNet_preprocessed/Task503_IBIS6M/nnUNetData_plans_v2.1_stage0'
    # folder_with_preprocessed_data = r'D:\OneDrive - stu.xjtu.edu.cn\my文档\longitudinal-representation-learning\code\nnUNet\DataSet\nnUNet_preprocessed\Task503_IBIS6M\nnUNetData_plans_v2.1_stage0'
    # unpack_dataset(folder_with_preprocessed_data)

    data_aug_params = np.load('./data_npy/503_dap_fold0.npy', allow_pickle=True).item()
    deep_supervision_scales = [[1, 1, 1], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125],
                               [0.0625, 0.0625, 0.0625]]
    pin_memory = True
    # tr_gen_501, val_gen_501 = get_moreDA_augmentation(dl_tr_501, dl_val_501,
    #                                           patch_size,
    #                                           data_aug_params,
    #                                           deep_supervision_scales=deep_supervision_scales,
    #                                           pin_memory=pin_memory, use_nondetMultiThreadedAugmenter=False)
    tr_gen_502, val_gen_502 = get_moreDA_augmentation(dl_tr_502, dl_val_502,
                                              patch_size,
                                              data_aug_params,
                                              deep_supervision_scales=deep_supervision_scales,
                                              pin_memory=pin_memory, use_nondetMultiThreadedAugmenter=False)
    # tr_gen_503, val_gen_503 = get_moreDA_augmentation(dl_tr_503, dl_val_503,
    #                                           patch_size,
    #                                           data_aug_params,
    #                                           deep_supervision_scales=deep_supervision_scales,
    #                                           pin_memory=pin_memory, use_nondetMultiThreadedAugmenter=False)
    tr_gen_504, val_gen_504 = get_moreDA_augmentation(dl_tr_504, dl_val_504,
                                              patch_size,
                                              data_aug_params,
                                              deep_supervision_scales=deep_supervision_scales,
                                              pin_memory=pin_memory, use_nondetMultiThreadedAugmenter=False)
    tr_gen_505, val_gen_505 = get_moreDA_augmentation(dl_tr_505, dl_val_505,
                                              patch_size,
                                              data_aug_params,
                                              deep_supervision_scales=deep_supervision_scales,
                                              pin_memory=pin_memory, use_nondetMultiThreadedAugmenter=False)


    batch_dice = False
    loss = DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
    ds_loss_weights = np.array([0.53333333, 0.26666667, 0.13333333, 0.06666667, 0.])
    loss = MultipleOutputLoss2(loss, ds_loss_weights)

    for i, param in enumerate(network.parameters()):
        # print(i)
        if (i > layer2) or (layer1 < i < 40):  # 冻结decoder
            param.requires_grad = False
        else:
            param.requires_grad = True
    optimizer_encoder = optim.SGD(filter(lambda p: p.requires_grad, network.parameters()), args.learning_rate, weight_decay=3e-5,
                                momentum=0.99, nesterov=True)
    for i, param in enumerate(network.parameters()):
        # print(i)
        if (i > layer2) or (layer1 < i < 40):  # 冻结encoder
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer_decoder1 = optim.SGD(filter(lambda p: p.requires_grad, network.parameters()), args.learning_rate2, weight_decay=3e-5,
                                momentum=0.99, nesterov=True)
    optimizer_decoder2 = optim.SGD(filter(lambda p: p.requires_grad, network.parameters()), args.learning_rate2, weight_decay=3e-5,
                                momentum=0.99, nesterov=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=100, gamma=0.5)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_decoder1, step_size=100, gamma=0.5)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer_decoder2, step_size=100, gamma=0.5)


    long_len = 500
    amp_grad_scaler = GradScaler()
    amp_grad_scaler2 = GradScaler()
    amp_grad_scaler3 = GradScaler()
    w1 = args.w1
    w2 = args.w2
    for epoch in range(epochs):
        network.train()
        with tqdm(total=long_len, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            tr_iter_list = [tr_gen_502, tr_gen_504, tr_gen_505]


            for num_itr in range(long_len//batch_size):
                # Randomly choosing meta train and meta test domains
                domain_list = np.random.permutation(3)
                meta_train_domain_list = domain_list[:2]
                meta_test_domain_list = domain_list[2]

                meta_train_imgs = []
                meta_train_labels = []


                data_dict = next(tr_iter_list[meta_train_domain_list[0]])
                data = data_dict['data']
                target = data_dict['target']
                data = maybe_to_torch(data)
                target = maybe_to_torch(target)
                if torch.cuda.is_available():
                    data = to_cuda(data, gpu_id=args.gpu)
                    target = to_cuda(target, gpu_id=args.gpu)
                meta_train_imgs.append(data)
                meta_train_labels.append(target)

                data_dict = next(tr_iter_list[meta_train_domain_list[1]])
                data = data_dict['data']
                target = data_dict['target']
                data = maybe_to_torch(data)
                target = maybe_to_torch(target)
                if torch.cuda.is_available():
                    data = to_cuda(data, gpu_id=args.gpu)
                    target = to_cuda(target, gpu_id=args.gpu)
                meta_train_imgs.append(data)
                meta_train_labels.append(target)


                data_dict = next(tr_iter_list[meta_test_domain_list])
                data = data_dict['data']
                target = data_dict['target']
                data = maybe_to_torch(data)
                target = maybe_to_torch(target)
                if torch.cuda.is_available():
                    data = to_cuda(data, gpu_id=args.gpu)
                    target = to_cuda(target, gpu_id=args.gpu)

                imgs = torch.cat((meta_train_imgs[0], meta_train_imgs[1]), dim=0)
                labels = [torch.cat((meta_train_labels[0][i], meta_train_labels[1][i]), dim=0) for i in range(5)]


                ###############################Support set#######################################################
                orien_oride = network.state_dict()
                for i, param in enumerate(network.parameters()):
                    # print(i)
                    if (i > layer2) or (layer1 < i < 40):  # 冻结encoder
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                optimizer_decoder1.zero_grad()
                with autocast():
                    output, _ = network(data)
                    l_decoder1 = loss(output, target)

                # l_decoder1.backward()
                amp_grad_scaler.scale(l_decoder1).backward()
                amp_grad_scaler.unscale_(optimizer_decoder1)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, network.parameters()), 12)
                amp_grad_scaler.step(optimizer_decoder1)
                amp_grad_scaler.update()
                # optimizer_decoder1.step()


                ###############################Query set 1#######################################################
                for i, param in enumerate(network.parameters()):
                    # print(i)
                    if (i > layer2) or (layer1 < i < 40):  # 冻结decoder
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                optimizer_encoder.zero_grad()
                with autocast():
                    output, latent = network(imgs)
                    del imgs
                    l_encoder = loss(output, labels)  # tensor(1.6946, device='cuda:0', grad_fn=<AddBackward0>)

                    l_inter = []
                    l_intra = []
                    for i in range(5):
                        # i=0
                        latent1of5 = latent[i]
                        # pdb.set_trace()
                        labels10f5 = labels[4 - i]
                        B, C, H, W, D = latent1of5.shape
                        latent1of5_reshape = latent1of5.reshape(B, C, -1)
                        labels10f5_reshape = labels10f5.reshape(B, 1, -1)
                        latent1of5_reshape_csf = torch.where(labels10f5_reshape == 1, latent1of5_reshape, 0)
                        latent1of5_reshape_gm = torch.where(labels10f5_reshape == 2, latent1of5_reshape, 0)
                        latent1of5_reshape_wm = torch.where(labels10f5_reshape == 3, latent1of5_reshape, 0)

                        num_csf = torch.zeros(B)
                        num_gm = torch.zeros(B)
                        num_wm = torch.zeros(B)
                        for b in range(B):
                            num_csf[b] = (labels10f5_reshape == 1)[b].sum()
                            num_gm[b] = (labels10f5_reshape == 2)[b].sum()
                            num_wm[b] = (labels10f5_reshape == 3)[b].sum()

                        f_csf = latent1of5_reshape_csf.sum(dim=-1) / num_csf.unsqueeze(-1).cuda(args.gpu)
                        f_gm = latent1of5_reshape_gm.sum(dim=-1) / num_gm.unsqueeze(-1).cuda(args.gpu)
                        f_wm = latent1of5_reshape_wm.sum(dim=-1) / num_wm.unsqueeze(-1).cuda(args.gpu)

                        l_inter.append((cossim(f_csf, f_gm) + cossim(f_csf, f_wm) + cossim(f_wm, f_gm))/3)  # 希望它大

                        f_csf_d1 = f_csf[:2, :]
                        f_gm_d1 = f_gm[:2, :]
                        f_wm_d1 = f_wm[:2, :]

                        f_csf_d2 = f_csf[2:4, :]
                        f_gm_d2 = f_gm[2:4, :]
                        f_wm_d2 = f_wm[2:4, :]

                        l_intra.append((cossim(f_csf_d1,f_csf_d2) + cossim(f_gm_d1, f_gm_d2) + cossim(f_wm_d1, f_wm_d2))/3)  #
                    # pdb.set_trace()
                    L_inter = sum(l_inter) / len(l_inter)
                    L_intra = sum(l_intra) / len(l_intra)
                    l_encoder = l_encoder + w1 * L_inter - w2 * L_intra
                # l_encoder.backward()
                amp_grad_scaler2.scale(l_encoder).backward()
                amp_grad_scaler2.unscale_(optimizer_encoder)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, network.parameters()), 12)
                amp_grad_scaler2.step(optimizer_encoder)
                amp_grad_scaler2.update()
                # optimizer_encoder.step()
                del labels

                # newen_newde = network.state_dict()
                # for i in range(98):
                #     if (i > layer2) or (layer1 < i < 40):  # 筛选decoder
                #         newen_newde[layers[i]] = orien_oride[layers[i]]
                # network.load_state_dict(newen_newde)
                # # newen_oride

                ###############################Query set 2#######################################################
                for i, param in enumerate(network.parameters()):
                    # print(i)
                    if (i > layer2) or (layer1 < i < 40):  # 冻结encoder
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                optimizer_decoder2.zero_grad()
                with autocast():
                    output, _ = network(data)
                    del data
                    l_decoder2 = loss(output, target)

                # # l_decoder2.backward()
                # amp_grad_scaler3.scale(l_decoder2).backward()
                # amp_grad_scaler3.unscale_(optimizer_decoder2)
                # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, network.parameters()), 12)
                # amp_grad_scaler3.step(optimizer_decoder2)
                # amp_grad_scaler3.update()
                # # optimizer_decoder2.step()

                optimizer_decoder2.zero_grad()
                l_decoder2.backward()

                newen_newde = network.state_dict()
                for i in range(98):
                    if (i > layer2) or (layer1 < i < 40):  # 筛选decoder
                        newen_newde[layers[i]] = orien_oride[layers[i]]
                network.load_state_dict(newen_newde)
                # newen_oride

                optimizer_decoder2.step()
                # newen_newde
                del target



                pbar.set_postfix(**{'l_decoder1': l_decoder1.item(),'l_intra': L_intra.item(),'l_inter': L_inter.item(), 'l_encoder': l_encoder.item(), 'l_decoder2': l_decoder2.item()})

                pbar.update(2)
            torch.save(network.state_dict(), f'./checkpoints/{model_name}_latest.pth')

            if (epoch+1) % 100 == 0:
                # torch.save(network, f'./checkpoints/metatrain_{global_step}.pth')
                torch.save(network.state_dict(), f'./checkpoints/{model_name}_{epoch+1}.pth')

        scheduler.step()
        scheduler2.step()
        scheduler3.step()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    torch.manual_seed(14)
    if device.type == 'cuda':
        torch.cuda.manual_seed(14)

    train_net(args)