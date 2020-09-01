#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import math
import json
import numpy as np
from tqdm import tqdm
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from util import transform_point_cloud, npmat2euler, quat2mat
from detect_flow import view_pointclouds

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def cycle_consistency(rotation_ab, translation_ab, rotation_ba, translation_ba):
    batch_size = rotation_ab.size(0)
    identity = torch.eye(3, device=rotation_ab.device).unsqueeze(0).repeat(batch_size, 1, 1)
    return F.mse_loss(torch.matmul(rotation_ab, rotation_ba), identity) + F.mse_loss(translation_ab, -translation_ba)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def pairwise_distance(src, tgt):
    inner = -2 * torch.matmul(src.transpose(2, 1).contiguous(), tgt)
    xx = torch.sum(src ** 2, dim=1, keepdim=True)
    yy = torch.sum(tgt ** 2, dim=1, keepdim=True)
    distances = xx.transpose(2, 1).contiguous() + inner + yy
    return torch.sqrt(distances + 1e-8)
    # return distances

def knn(x, k=20):
    x = x.view(*x.size()[:3])
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = distance.sort(descending=True, dim=-1)[1]  # (batch_size, num_points, k)
    return idx[:, :, :k]

def get_graph_feature(x, idx=None, k=20):
    # x = x.squeeze(-1)
    x = x.view(*x.size()[:3])
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature

def cycle_consistency(rotation_ab, translation_ab, rotation_ba, translation_ba):
    batch_size = rotation_ab.size(0)
    identity = torch.eye(3, device=rotation_ab.device).unsqueeze(0).repeat(batch_size, 1, 1)
    return F.mse_loss(torch.matmul(rotation_ab, rotation_ba), identity) + F.mse_loss(translation_ab, -translation_ba)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LDGCNN(nn.Module):
    def __init__(self, n_emb_dims=512):
        super(LDGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(384, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(320, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(n_emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = x.unsqueeze(-1)
        edge_feature = get_graph_feature(x)
        net = F.relu(self.bn1(self.conv1(edge_feature)))
        # (bs, 64, num_points, 1)
        net = net.max(dim=-1, keepdim=True)[0]
        net1 = net
        idx = knn(net)
        # (bs, 128, np, k)
        edge_feature = get_graph_feature(net, idx)
        # (bs, 64, num_points, k)
        net = F.relu(self.bn2(self.conv2(edge_feature)))
        # (bs, 64, num_points, 1)
        net = net.max(dim=-1, keepdim=True)[0]
        net2 = net
        idx = knn(net)
        # (bs, 128, num_points, 1)
        net = torch.cat([net1, net2], dim=1)
        # (bs, 256, np, k)
        edge_feature = get_graph_feature(net, idx)
        net = F.relu(self.bn3(self.conv3(edge_feature)))
        net = net.max(dim=-1, keepdim=True)[0]
        net3 = net
        idx = knn(net)
        # (bs, 195, np, 1)
        net = torch.cat([net1, net2, net3], dim=1)
        # (bs, 384, np, k)
        edge_feature = get_graph_feature(net, idx)
        # (bs, 128, np, k)
        net = F.relu(self.bn4(self.conv4(edge_feature)))
        # (bs, 128, np, 1)
        net = net.max(dim=-1, keepdim=True)[0]
        net4 = net
        # (bs, 320, np, 1)
        net = torch.cat([net1, net2, net3, net4], dim=1)
        # (bs, 512, np, 1)
        net = F.relu(self.bn5(self.conv5(net)))
        # (bs, 512, np)
        net = net.squeeze(-1)
        return net

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.n_ff_dims = args.n_ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.n_emb_dims)
        ff = PositionwiseFeedForward(self.n_emb_dims, self.n_ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.n_emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.n_emb_dims, c(attn), c(attn), c(ff), self.dropout),
                                            self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding

class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.dims = args.n_emb_dims
        self.n_keypoints = args.n_keypoints
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.my_iter = torch.ones(1)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 1, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(1)

    def sinkhorn(self, scores, n_iters):
        # scores: (bs, k, np)
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        # scores: (bs, k+1, np)
        scores = zero_pad(scores)
        for i in range(n_iters):
            # row normalization
            scores = torch.cat((
                    scores[:, :-1, :] - (torch.logsumexp(scores[:, :-1, :], dim=2, keepdim=True)),
                    scores[:, -1, None, :]),  # Don't normalize last row
                dim=1)
            # col normalization
            scores = torch.cat((
                scores[:, :, :-1] - (torch.logsumexp(scores[:, :, :-1], dim=1, keepdim=True)),
                scores[:, :, -1, None]),  # Don't normalize last row
                dim=2)
        # (bs, k, np)
        scores = scores[:, :-1, :-1]
        return scores

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size, d_k, num_points_k = src_embedding.size()
        num_points = tgt.shape[2]
        temperature = input[4].view(batch_size, 1, 1)
        # (bs, np, np)
        dists = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        affinity = dists / temperature
        # (bs, np, np)
        log_perm_matrix = self.sinkhorn(affinity, n_iters=5)
        # (bs, k, np)
        perm_matrix = torch.exp(log_perm_matrix)
        perm_matrix_norm = perm_matrix / (torch.sum(perm_matrix, dim=2, keepdim=True) + 1e-8)
        # (bs, 3, np)
        src_corr = torch.matmul(tgt, perm_matrix_norm.transpose(2, 1).contiguous())
        # (bs, dim, np)
        src_corr_embedding = torch.matmul(tgt_embedding, perm_matrix_norm.transpose(2, 1).contiguous())
        embedding = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_corr_embedding) / math.sqrt(d_k)
        # (bs, 1, np)
        embedding = torch.diagonal(embedding, dim1=-2, dim2=-1).unsqueeze(1)
        # embedding = torch.cat([src_embedding, src_corr_embedding], dim=1)
        # (bs, 1, np)
        x = F.relu(self.bn1(self.conv1(embedding)))
        x = F.relu(self.bn2(self.conv2(x)))
        corr_scores = x.repeat(1, self.n_keypoints, 1)
        temperature = temperature.view(batch_size, 1)
        corr_scores = corr_scores.view(batch_size * self.n_keypoints, num_points)
        temperature = temperature.repeat(1, self.n_keypoints, 1).view(-1, 1)
        corr_scores = F.gumbel_softmax(corr_scores, tau=temperature, hard=True)
        # (bs, k, np)
        corr_scores = corr_scores.view(batch_size, self.n_keypoints, num_points)
        src_k = torch.matmul(corr_scores, src.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()
        src_corr_k = torch.matmul(corr_scores, src_corr.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()
        src_centered = src_k - src_k.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr_k - src_corr_k.mean(dim=2, keepdim=True)
        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous()).cpu()
        R = []
        for i in range(src.size(0)):
            try:
                u, s, v = torch.svd(H[i])
            except:
                print(H[i])
            r = torch.matmul(v, u.transpose(1, 0)).contiguous()
            r_det = torch.det(r).item()
            diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                              [0, 1.0, 0],
                                              [0, 0, r_det]]).astype('float32')).to(v.device)
            r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
            R.append(r)
        R = torch.stack(R, dim=0).cuda()
        t = torch.matmul(-R, src_k.mean(dim=2, keepdim=True)) + src_corr_k.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3), perm_matrix_norm

class MatchNet(nn.Module):
    def __init__(self, args):
        super(MatchNet, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.num_keypoints = args.n_keypoints
        self.num_subsampled_points = args.n_subsampled_points
        self.n_iters = args.n_iters
        for i in range(self.n_iters):
            layer = LDGCNN(n_emb_dims=self.n_emb_dims)
            attn = Transformer(args=args)
            head = SVDHead(args=args)
            self.add_module('emb_nn_{}'.format(i), layer)
            self.add_module('attention_{}'.format(i), attn)
            self.add_module('head_{}'.format(i), head)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        temp = input[2]
        i = input[3]
        emb_nn = getattr(self, 'emb_nn_{}'.format(i))
        src_embedding = emb_nn(src)
        tgt_embedding = emb_nn(tgt)
        attn = getattr(self, 'attention_{}'.format(i))
        src_embedding_p, tgt_embedding_p = attn(src_embedding, tgt_embedding)
        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p
        head = getattr(self, 'head_{}'.format(i))
        rotation_ab, translation_ab, scores = head(src_embedding, tgt_embedding, src, tgt, temp)
        return rotation_ab, translation_ab, scores

class HMNet(nn.Module):
    def __init__(self, args):
        super(HMNet, self).__init__()
        self.num_iters = args.n_iters
        self.logger = Logger(args)
        self.match_net = MatchNet(args)
        self.model_path = args.model_path
        self.discount_factor = args.discount_factor
        if self.model_path is not '':
            self.load(self.model_path)
        if torch.cuda.device_count() > 1:
            self.match_net = nn.DataParallel(self.match_net)

    def compute_loss(self, src, tgt, scores, rotation_ab, translation_ab):
        src_gt = transform_point_cloud(src, rotation_ab, translation_ab)
        # view_pointclouds(src_k_gt.squeeze(0).cpu().detach().numpy().T, tgt.squeeze(0).cpu().detach().numpy().T)
        dists = pairwise_distance(src_gt, tgt)
        # (bs, k, np)
        sort_distance, sort_id = torch.sort(dists, dim=-1)
        # (bs, k, 1) 距离最近的id 设阈值小于0.1的为关键点
        TD = sort_id[:, :, 0, None]
        # (bs, k, 1)
        nearest_dist = sort_distance[:, :, 0, None]
        # (bs, k, 1)
        S = torch.gather(- torch.log(scores + 1e-8), index=TD, dim=-1)
        S_zeros = torch.zeros_like(S)
        # 超参需要手动调整
        ind_S = torch.where(nearest_dist > 0.08, S_zeros, S)
        S_loss = torch.mean(ind_S)
        return S_loss

    def forward(self, *input):
        rotation_ab, translation_ab, scores = self.match_net(*input)
        return rotation_ab, translation_ab, scores

    def _train_one_batch(self, src, tgt, rotation_ab, translation_ab, opt, temp):
        opt.zero_grad()
        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        total_loss = 0
        temp = torch.tensor(temp).cuda().repeat(batch_size)
        for i in range(self.num_iters):
            rotation_ab_pred_i, translation_ab_pred_i, scores = self.forward(src, tgt, temp, i)
            # 期望位姿
            res_rotation_ab = torch.matmul(rotation_ab, rotation_ab_pred.transpose(2, 1))
            res_translation_ab = translation_ab - torch.matmul(res_rotation_ab,
                                                               translation_ab_pred.unsqueeze(2)).squeeze(2)
            # 累计位姿
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i
            # 位姿loss
            loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity)\
                    + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor ** i
            # 熵值loss
            entropy_loss = self.compute_loss(src, tgt, scores, res_rotation_ab, res_translation_ab,
                                             ) * self.discount_factor ** i
            total_loss = total_loss + loss + entropy_loss * 0.5
            # 这个时候点云才会变
            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)
        total_loss.backward()
        opt.step()
        return total_loss.item(), rotation_ab_pred, translation_ab_pred

    def _test_one_batch(self, src, tgt, rotation_ab, translation_ab, temp):
        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)
        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        total_loss = 0
        temp = torch.tensor(temp).cuda().repeat(batch_size)
        for i in range(self.num_iters):
            rotation_ab_pred_i, translation_ab_pred_i, scores = self.forward(src, tgt, temp, i)
            # 期望位姿
            res_rotation_ab = torch.matmul(rotation_ab, rotation_ab_pred.transpose(2, 1))
            res_translation_ab = translation_ab - torch.matmul(res_rotation_ab,
                                                               translation_ab_pred.unsqueeze(2)).squeeze(2)
            # 累计位姿
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i
            # 位姿loss
            loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                    + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor ** i
            # 熵值loss
            entropy_loss = self.compute_loss(src, tgt, scores, res_rotation_ab, res_translation_ab,
                                             ) * self.discount_factor ** i
            total_loss = total_loss + loss + entropy_loss * 0.5
            # 这个时候点云才会变
            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)
        return total_loss.item(), rotation_ab_pred, translation_ab_pred

    def _train_one_epoch(self, epoch, train_loader, opt, temp):
        self.train()
        total_loss = 0
        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        eulers_ab = []
        num_examples = 0
        for data in tqdm(train_loader):
            src, tgt, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = [d.cuda()
                                                                                                      for d in data]
            loss, rotation_ab_pred, translation_ab_pred = self._train_one_batch(src, tgt, rotation_ab, translation_ab,
                                                                                opt, temp)
            batch_size = src.size(0)
            num_examples += batch_size
            total_loss = total_loss + loss * batch_size
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.cpu().numpy())
        avg_loss = total_loss / num_examples
        # (num_examples, 3, 3)
        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        eulers_ab = np.degrees(np.concatenate(eulers_ab, axis=0))
        eulers_ab_pred = npmat2euler(rotations_ab_pred)
        r_ab_mse = np.mean((eulers_ab - eulers_ab_pred) ** 2)
        r_ab_rmse = np.sqrt(r_ab_mse)
        r_ab_mae = np.mean(np.abs(eulers_ab - eulers_ab_pred))
        rot = np.matmul(rotations_ab_pred.transpose(0, 2, 1), rotations_ab)
        rot_trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
        residual_rotdeg = np.arccos(np.clip(0.5 * (rot_trace - 1), a_min=-1.0, a_max=1.0)) * 180.0 / np.pi
        residual_rotdeg = np.mean(residual_rotdeg)
        t_ab_mse = np.mean((translations_ab - translations_ab_pred) ** 2)
        t_ab_rmse = np.sqrt(t_ab_mse)
        t_ab_mae = np.mean(np.abs(translations_ab - translations_ab_pred))
        r_ab_r2_score = r2_score(eulers_ab, eulers_ab_pred)
        t_ab_r2_score = r2_score(translations_ab, translations_ab_pred)
        info = {'arrow': 'A->B',
                'epoch': epoch,
                'stage': 'train',
                'loss': avg_loss,
                'r_ab_mse': r_ab_mse,
                'r_ab_rmse': r_ab_rmse,
                'r_ab_mae': r_ab_mae,
                't_ab_mse': t_ab_mse,
                't_ab_rmse': t_ab_rmse,
                't_ab_mae': t_ab_mae,
                'r_ab_r2_score': r_ab_r2_score,
                't_ab_r2_score': t_ab_r2_score,
                'residual_rotdeg': residual_rotdeg,
                'temperature': temp}
        self.logger.write(info)
        return info

    def _test_one_epoch(self, epoch, test_loader, temp):
        self.eval()
        total_loss = 0
        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        eulers_ab = []
        num_examples = 0
        for data in tqdm(test_loader):
            src, tgt, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = [d.cuda()
                                                                                                      for d in data]
            loss, rotation_ab_pred, translation_ab_pred = self._test_one_batch(src, tgt, rotation_ab, translation_ab,
                                                                               temp)
            batch_size = src.size(0)
            num_examples += batch_size
            total_loss = total_loss + loss * batch_size
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.cpu().numpy())
        avg_loss = total_loss / num_examples
        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        eulers_ab = np.degrees(np.concatenate(eulers_ab, axis=0))
        eulers_ab_pred = npmat2euler(rotations_ab_pred)
        r_ab_mse = np.mean((eulers_ab - eulers_ab_pred) ** 2)
        r_ab_rmse = np.sqrt(r_ab_mse)
        r_ab_mae = np.mean(np.abs(eulers_ab - eulers_ab_pred))
        rot = np.matmul(rotations_ab_pred.transpose(0, 2, 1), rotations_ab)
        rot_trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
        residual_rotdeg = np.arccos(np.clip(0.5 * (rot_trace - 1), a_min=-1.0, a_max=1.0)) * 180.0 / np.pi
        residual_rotdeg = np.mean(residual_rotdeg)
        t_ab_mse = np.mean((translations_ab - translations_ab_pred) ** 2)
        t_ab_rmse = np.sqrt(t_ab_mse)
        t_ab_mae = np.mean(np.abs(translations_ab - translations_ab_pred))
        r_ab_r2_score = r2_score(eulers_ab, eulers_ab_pred)
        t_ab_r2_score = r2_score(translations_ab, translations_ab_pred)
        info = {'arrow': 'A->B',
                'epoch': epoch,
                'stage': 'test',
                'loss': avg_loss,
                'r_ab_mse': r_ab_mse,
                'r_ab_rmse': r_ab_rmse,
                'r_ab_mae': r_ab_mae,
                't_ab_mse': t_ab_mse,
                't_ab_rmse': t_ab_rmse,
                't_ab_mae': t_ab_mae,
                'r_ab_r2_score': r_ab_r2_score,
                't_ab_r2_score': t_ab_r2_score,
                'residual_rotdeg': residual_rotdeg,
                'temperature': temp}
        self.logger.write(info)
        return info

    def save(self, path):
        if torch.cuda.device_count() > 1:
            torch.save(self.match_net.module.state_dict(), path)
        else:
            torch.save(self.match_net.state_dict(), path)

    def load(self, path):
        self.match_net.load_state_dict(torch.load(path))
        print("loading successful!")


class Logger:
    def __init__(self, args):
        self.path = 'checkpoints/' + args.exp_name
        self.fw = open(self.path + '/log', 'a')
        self.fw.write(str(args))
        self.fw.write('\n')
        self.fw.flush()
        print(str(args))
        with open(os.path.join(self.path, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def write(self, info):
        arrow = info['arrow']
        epoch = info['epoch']
        stage = info['stage']
        loss = info['loss']
        r_ab_mse = info['r_ab_mse']
        r_ab_rmse = info['r_ab_rmse']
        r_ab_mae = info['r_ab_mae']
        t_ab_mse = info['t_ab_mse']
        t_ab_rmse = info['t_ab_rmse']
        t_ab_mae = info['t_ab_mae']
        r_ab_r2_score = info['r_ab_r2_score']
        t_ab_r2_score = info['t_ab_r2_score']
        residual_rotdeg = info['residual_rotdeg']
        temperature = info['temperature']
        text = '%s:: Stage: %s, Epoch: %d, Loss: %f, Rot_MSE: %f, Rot_RMSE: %f, ' \
               'Rot_MAE: %f, Rot_R2: %f, Trans_MSE: %f, ' \
               'Trans_RMSE: %f, Trans_MAE: %f, Trans_R2: %f, Rot_deg: %f, temperature: %f\n' % \
               (arrow, stage, epoch, loss, r_ab_mse, r_ab_rmse, r_ab_mae,
                r_ab_r2_score, t_ab_mse, t_ab_rmse, t_ab_mae, t_ab_r2_score, residual_rotdeg, temperature)
        self.fw.write(text)
        self.fw.flush()
        print(text)

    def close(self):
        self.fw.close()

if __name__ == '__main__':
    print('hello world')
