# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

# Modified by: Daquan Zhou

'''
- resize_pos_embed: resize position embedding
- load_for_transfer_learning: load pretrained paramters to model in transfer learning
- get_mean_and_std: calculate the mean and std value of dataset.
- msr_init: net parameter initialization.
- progress_bar: progress bar mimic xlua.progress.
'''

import os
import sys
import time
import torch
import math

import torch.nn as nn
import torch.nn.init as init
import logging
import os
from collections import OrderedDict
import torch.nn.functional as F

import csv
import pandas as pd

_logger = logging.getLogger(__name__)

def resize_pos_embed(posemb, posemb_new): # example: 224:(14x14+1)-> 384: (24x24+1)
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]  # posemb_tok is for cls token, posemb_grid for the following tokens
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))     # 14
    gs_new = int(math.sqrt(ntok_new))             # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic') # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)   # [1, dim, 24, 24] -> [1, 24*24, dim]
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)   # [1, 24*24+1, dim]
    return posemb

def resize_pos_embed_cait(posemb, posemb_new): # example: 224:(14x14+1)-> 384: (24x24+1)
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    posemb_grid = posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))     # 14
    gs_new = int(math.sqrt(ntok_new))             # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic') # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)   # [1, dim, 24, 24] -> [1, 24*24, dim]
    return posemb_grid


def resize_pos_embed_nocls(posemb, posemb_new): # example: 224:(14x14+1)-> 384: (24x24+1)
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    gs_old = posemb.shape[1]     # 14
    gs_new = posemb_new.shape[1]             # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb
    posemb_grid = posemb_grid.permute(0, 3, 1, 2)  # [1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic') # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1)   # [1, dim, 24, 24]->[1, 24, 24, dim]
    return posemb_grid


def load_state_dict(checkpoint_path,model, use_ema=False, num_classes=1000, no_pos_embed=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        if num_classes != 1000:
            # completely discard fully connected for all other differences between pretrained and created model
            del state_dict['head' + '.weight']
            del state_dict['head' + '.bias']
            old_aux_head_weight = state_dict.pop('aux_head.weight', None)
            old_aux_head_bias = state_dict.pop('aux_head.bias', None)
        if not no_pos_embed:
            old_posemb = state_dict['pos_embed']
            if model.pos_embed.shape != old_posemb.shape:  # need resize the position embedding by interpolate
                if len(old_posemb.shape)==3:
                    if int(math.sqrt(old_posemb.shape[1]))**2==old_posemb.shape[1]:
                        new_posemb = resize_pos_embed_cait(old_posemb, model.pos_embed)
                    else:
                        new_posemb = resize_pos_embed(old_posemb, model.pos_embed)
                elif len(old_posemb.shape)==4:
                    new_posemb = resize_pos_embed_nocls(old_posemb, model.pos_embed)
                state_dict['pos_embed'] = new_posemb

        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_for_transfer_learning(model, checkpoint_path, use_ema=False, strict=True, num_classes=1000):
    state_dict = load_state_dict(checkpoint_path, model, use_ema, num_classes)
    model.load_state_dict(state_dict, strict=strict)

def load_for_probing(model, checkpoint_path, use_ema=False, strict=False, num_classes=19167):
    state_dict = load_state_dict(checkpoint_path, model, use_ema, num_classes=19167, no_pos_embed=True)
    info=model.load_state_dict(state_dict, strict=strict)
    print(info)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def gist_multi_level_loss(gist_list, cls_list):
    loss = 0
    for i in range(len(gist_list)):
        loss += F.mse_loss(gist_list[i] , cls_list[i])
    return loss



def check_and_initialize_csv(directory_path, csv_path, headers):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        with open(csv_path, 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def update_dataset_metric(csv_path, dataset, best_metric, headers):
    rows = []
    with open(csv_path, 'r', encoding='utf8') as f:
        for row in csv.reader(f):
            rows.append(row)
    
    # If only header exists, append an empty row
    if len(rows) == 1:
        rows.append([None] * len(headers))
    
    # Update the last row
    last_row = rows[-1]
    last_row[headers.index(dataset)] = best_metric
    # Write back the rows
    with open(csv_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def update_dataset_metric_adv(csv_path, dataset, best_metric, headers, is_adv=False):
    rows = []
    with open(csv_path, 'r', encoding='utf8') as f:
        for row in csv.reader(f):
            rows.append(row)
    
    # If only header exists, append an empty row
    if len(rows) == 1:
        rows.append([None] * len(headers))
        rows.append([None] * len(headers))
    
    # Update the first row
    first_row, last_row = rows[-2], rows[-1]
    if is_adv:
        last_row[headers.index(dataset)] = best_metric
    else:
        first_row[headers.index(dataset)] = best_metric

    # Write back the rows
    with open(csv_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def update_fs_dataset_metric(csv_path, dataset, best_metric, headers, shot, seed):
    rows = []
    shot_index = headers.index('SHOT')  # Assuming 'SHOT' is a header
    seed_index = headers.index('SEED')  # Assuming 'SEED' is a header
    dataset_index = headers.index(dataset)
    updated = False

    # Check if the CSV file exists and read its content
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf8') as f:
            rows = list(csv.reader(f))

    # If the CSV is empty, initialize with headers
    if not rows:
        rows.append(headers)

    # Look for an existing row with the same SHOT and SEED
    for row in rows[1:]:  # Skip the header row
        if row[shot_index] == str(shot) and row[seed_index] == str(seed):
            row[dataset_index] = best_metric  # Update the existing row
            updated = True
            break

    # If no existing row is found, create and append a new row
    if not updated:
        new_row = [None] * len(headers)
        new_row[shot_index] = str(shot)
        new_row[seed_index] = str(seed)
        new_row[dataset_index] = best_metric
        rows.append(new_row)

    # Write the rows back to the CSV file
    with open(csv_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def get_weight_distillation_loss(student_weight_list, teacher_weight_list, loss_scaler=0.1):
    loss = 0
    for i in range(len(student_weight_list)):
        loss += F.mse_loss(student_weight_list[i].mean(dim=-1), teacher_weight_list[i].mean(dim=-1))
    return loss * loss_scaler


# def get_weight_distillation_loss_by_interpolate(student_weight_list, teacher_weight_list, loss_scaler=0.1):
#     loss = 0
#     if teacher_weight_list[0].size() != student_weight_list[0].size():
#         teacher_weight_interpolate_size=teacher_weight_list[0].shape[-1]
#         for i in range(len(teacher_weight_list)):
#             student_weight = F.interpolate(student_weight_list[i].unsqueeze(0), size=(teacher_weight_interpolate_size,), mode='linear', align_corners=True).squeeze(0)
#             loss += F.mse_loss(student_weight, teacher_weight_list[i])
#     else:
#         for i in range(len(teacher_weight_list)):
#             loss += F.mse_loss(student_weight_list[i], teacher_weight_list[i])
#     return loss * loss_scaler

def get_weight_distillation_loss_by_interpolate(student_weight_list, teacher_weight_list, loss_scaler=1):
    loss = 0
    teacher_weight_interpolate_size=teacher_weight_list[0].shape[-1]
    for i in range(len(teacher_weight_list)):
        student_weight = F.interpolate(student_weight_list[i].unsqueeze(0), size=(teacher_weight_interpolate_size,), mode='linear', align_corners=True).squeeze(0)
        # loss += F.mse_loss(student_weight, teacher_weight_list[i])
        loss += F.kl_div(F.softmax(student_weight/5.0, dim=-1).log(), F.softmax(teacher_weight_list[i]/5.0, dim=-1), reduction='batchmean')
        # print(student_weight.size(), teacher_weight_list[i].size())
    return loss * loss_scaler


def Cross_Attention(q, k, v, num_heads=8):
    # 假设 q, k, v 的维度为 (in_dim, out_dim)，即 (seq_len, feature_dim)
    seq_len, feature_dim = q.shape
    # 确保特征维度可以被注意力头数量整除
    assert feature_dim % num_heads == 0, 'dim should be divisible by num_heads'
    head_dim = feature_dim // num_heads
    scale = head_dim ** -0.5
    # 重塑和置换 q, k, v 以适应多头注意力格式
    # 假设 batch_size = 1，因为没有提供这个维度
    q = q.reshape(1, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.reshape(1, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    v = v.reshape(1, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    # 计算注意力分数
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    attn = F.dropout(attn, 0.1, training=True)
    # 输出计算
    x = (attn @ v).transpose(1, 2).reshape(1, seq_len, feature_dim)
    return x
    
def cal_loss_cross_att_sim(student_weight, teacher_weight):
    t_s_out = Cross_Attention(teacher_weight, student_weight, student_weight)
    s_t_out = Cross_Attention(student_weight, teacher_weight, teacher_weight)
    return F.mse_loss(t_s_out, s_t_out)

def get_weight_distillation_loss_by_cross_att(student_weight_list, teacher_weight_list, loss_scaler=0.1):
    loss = 0
    if teacher_weight_list[0].size() != student_weight_list[0].size():
        teacher_weight_interpolate_size=teacher_weight_list[0].shape[-1]
        for i in range(len(teacher_weight_list)):
            student_weight = F.interpolate(student_weight_list[i].unsqueeze(0), size=(teacher_weight_interpolate_size,), mode='linear', align_corners=True).squeeze(0)
            loss += cal_loss_cross_att_sim(student_weight, teacher_weight_list[i])
    else:
        for i in range(len(teacher_weight_list)):
            loss += cal_loss_cross_att_sim(student_weight_list[i], teacher_weight_list[i])
    return loss * loss_scaler
