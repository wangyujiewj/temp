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
from util import transform_point_cloud, npmat2euler

def pairwise_distance(src, tgt):
    inner = -2 * torch.matmul(src.transpose(2, 1).contiguous(), tgt)
    xx = torch.sum(src ** 2, dim=1, keepdim=True)
    yy = torch.sum(tgt ** 2, dim=1, keepdim=True)
    distances = xx.transpose(2, 1).contiguous() + inner + yy
    return torch.sqrt(distances + 1e-8)

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

class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x

class GSS(nn.Module):
    def __init__(self, args):
        super(GSS, self).__init__()
        self.n_keypoints = args.n_keypoints
        self.dims = args.n_emb_dims
        self.proj = nn.Linear(self.dims, 1).cuda()
        # output -> (0,1)

    def forward(self, *input):
        embedding = input[0]
        points = input[1]
        batch_size = points.shape[0]
        num_points = points.shape[2]
        temperature = input[2].view(batch_size, 1, 1)
        assert (embedding.shape[2] == points.shape[2])
        # (bs, dim, num_points) -> (bs, n_keypoints, num_points)
        embedding = embedding.transpose(2, 1).contiguous()
        # (bs, k, np) i.i.d.
        scores = self.proj(embedding).repeat(1, 1, self.n_keypoints).transpose(2, 1).contiguous()
        temperature = temperature.view(batch_size, 1)
        scores = scores.view(batch_size * self.n_keypoints, num_points)
        temperature = temperature.repeat(1, self.n_keypoints, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, self.n_keypoints, num_points)
        new_points = torch.matmul(scores, points.transpose(2, 1).contiguous())
        new_embedding = torch.matmul(scores, embedding)
        return new_embedding.transpose(2, 1).contiguous(), new_points.transpose(2, 1).contiguous()

class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.my_iter = torch.ones(1)

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
        temperature = input[4].view(batch_size, 1, 1)
        # (bs, k, np)
        dists = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        affinity = dists / temperature
        log_perm_matrix = self.sinkhorn(affinity, n_iters=5)
        # (bs, k, np)
        perm_matrix = torch.exp(log_perm_matrix)
        perm_matrix_norm = perm_matrix / (torch.sum(perm_matrix, dim=2, keepdim=True) + 1e-8)
        # (bs, 3, k)
        weighted_tgt = torch.matmul(tgt, perm_matrix_norm.transpose(2, 1).contiguous())
        # (bs, k, 1)
        weights = torch.sum(perm_matrix, dim=-1, keepdim=True)
        # (bs, k, 1)
        weights_norm = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
        # (bs, 1, k)
        weights_norm = weights_norm.transpose(2, 1).contiguous()
        # (bs, 3, 1)
        src_centroid = torch.sum(torch.mul(src, weights_norm), dim=2, keepdim=True)
        tgt_centroid = torch.sum(torch.mul(weighted_tgt, weights_norm), dim=2, keepdim=True)
        src_centered = src - src_centroid
        src_corr_centered = weighted_tgt - tgt_centroid
        src_corr_centered = torch.mul(src_corr_centered, weights_norm)
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
        t = torch.matmul(-R, src_centroid) + tgt_centroid
        return R, t.view(batch_size, 3), perm_matrix_norm

class MatchNet(nn.Module):
    def __init__(self, args):
        super(MatchNet, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.num_keypoints = args.n_keypoints
        self.num_subsampled_points = args.n_subsampled_points
        self.n_iters = args.n_iters
        self.tgt_emb_nn = DGCNN(emb_dims=self.n_emb_dims)
        for i in range(self.n_iters):
            layer = DGCNN(emb_dims=self.n_emb_dims)
            self.add_module('src_emb_nn_{}'.format(i), layer)
            gss = GSS(args)
            self.add_module('sampling_{}'.format(i), gss)
        self.head = SVDHead(args=args)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        temp = input[2]
        i = input[3]
        tgt_embedding = self.tgt_emb_nn(tgt)
        src_emb_nn = getattr(self, 'src_emb_nn_{}'.format(i))
        src_embedding = src_emb_nn(src)
        sampling_layer = getattr(self, 'sampling_{}'.format(i))
        src_embedding_k, src_k = sampling_layer(src_embedding, src, temp)
        rotation_ab, translation_ab, scores = self.head(src_embedding_k, tgt_embedding, src_k, tgt, temp)
        return rotation_ab, translation_ab, src_k, scores

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
    def forward(self, *input):
        rotation_ab, translation_ab, src_k, scores = self.match_net(*input)
        return rotation_ab, translation_ab, src_k, scores

    def compute_loss(self, scores, src, rotation_ab, translation_ab, tgt):
        src_gt = transform_point_cloud(src, rotation_ab, translation_ab)
        # view_pointclouds(src_k_gt.squeeze(0).cpu().detach().numpy().T, tgt.squeeze(0).cpu().detach().numpy().T)
        dists = pairwise_distance(src_gt, tgt)
        # (bs, k, np)
        sort_distance, sort_id = torch.sort(dists, dim=-1)
        # (bs, k, 1) 距离最近的id 设阈值小于0.1的为关键点
        TD = sort_id[:, :, 0, None]
        # (bs, np, 1)
        nearest_dist = sort_distance[:, :, 0, None]
        # (bs, np, 1)
        S = torch.gather(-torch.log(scores + 1e-8), index=TD, dim=-1)
        S = torch.mean(S)
        return S

    def _train_one_batch(self, src, tgt, rotation_ab, translation_ab, opt, temp):
        opt.zero_grad()
        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)
        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        total_loss = 0
        temp = torch.tensor(temp).cuda().repeat(batch_size)
        for i in range(self.num_iters):
            rotation_ab_pred_i, translation_ab_pred_i, src_k, scores = self.forward(src, tgt, temp, i)
            # 残差位姿
            res_rotation_ab = torch.matmul(rotation_ab, rotation_ab_pred.transpose(2, 1))
            res_translation_ab = translation_ab - torch.matmul(res_rotation_ab,
                                                               translation_ab_pred.unsqueeze(2)).squeeze(2)
            # 累计位姿
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i
            # 熵值loss
            entropy_loss = self.compute_loss(scores, src_k, res_rotation_ab, res_translation_ab, tgt)
            total_loss = total_loss + entropy_loss
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
            rotation_ab_pred_i, translation_ab_pred_i, src_k, scores = self.forward(src, tgt, temp, i)
            # 残差位姿
            res_rotation_ab = torch.matmul(rotation_ab, rotation_ab_pred.transpose(2, 1))
            res_translation_ab = translation_ab - torch.matmul(res_rotation_ab,
                                                               translation_ab_pred.unsqueeze(2)).squeeze(2)
            # 累计位姿
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i
            # 熵值loss
            entropy_loss = self.compute_loss(scores, src_k, res_rotation_ab, res_translation_ab, tgt)
            total_loss = total_loss + entropy_loss
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
