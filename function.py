import torch
import torch.nn as nn
from util import transform_point_cloud
import numpy as np
class MCCLossFunc(nn.Module):
    def __init__(self):
        super(MCCLossFunc, self).__init__()
        return
    def forward(self, src, R_pred, t_pred, R_gt, t_gt, sigma_):
        batch_size = R_pred.size(0)
        sigma_ = sigma_.view(batch_size, 1)
        # (bs,)
        beta = (1 - torch.exp(-(0.5 * sigma_**(-2)))) ** (-1)
        src_pred = transform_point_cloud(src, R_pred, t_pred)
        src_gt = transform_point_cloud(src, R_gt, t_gt)
        # (bs, np)
        error = torch.sum((src_pred - src_gt)**2, dim=1)
        error = torch.sqrt(error)
        exp_error = torch.exp(-(error * 0.5 * sigma_**(-2)))
        one_tensor = exp_error.new_tensor(1).expand(batch_size,)
        mcc_loss = - torch.mean(exp_error, dim=-1) + one_tensor
        mcc_loss = mcc_loss * beta.squeeze(-1)
        mean_mcc_loss = torch.mean(mcc_loss, dim=0)
        return mean_mcc_loss

class MCCLossFuncv2(nn.Module):
    def __init__(self):
        super(MCCLossFuncv2, self).__init__()
        return
    def forward(self, src_k, R_gt, t_gt, src_corr, sigma_):
        batch_size = R_gt.size(0)
        sigma_ = sigma_.view(batch_size, 1)
        # (bs,)
        beta = (1 - torch.exp(-(0.5 * sigma_**(-2)))) ** (-1)
        src_gt = transform_point_cloud(src_k, R_gt, t_gt)
        # (bs, np)
        error = torch.sum((src_corr - src_gt)**2, dim=1)
        error = torch.sqrt(error)
        exp_error = torch.exp(-(error * 0.5 * sigma_**(-2)))
        one_tensor = exp_error.new_tensor(1).expand(batch_size,)
        mcc_loss = - torch.mean(exp_error, dim=-1) + one_tensor
        mcc_loss = mcc_loss * beta.squeeze(-1)
        mean_mcc_loss = torch.mean(mcc_loss, dim=0)
        return mean_mcc_loss

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()
        return
    def forward(self, sampling_scores, src, tgt, rotation_ab, translation_ab):
        bs, k, num_points = sampling_scores.size()
        src_corr = transform_point_cloud(src, rotation_ab, translation_ab)
        inner = -2 * torch.matmul(src_corr.transpose(2, 1).contiguous(), tgt)
        xx = torch.sum(src_corr ** 2, dim=1, keepdim=True)
        yy = torch.sum(tgt ** 2, dim=1, keepdim=True)
        distance = xx.transpose(2, 1).contiguous() + inner + yy
        nearst_dist, _ = distance.sort(dim=-1)
        # (bs, np)
        nearst_dist = nearst_dist [:,:,0].squeeze(-1)
        idx = nearst_dist.sort(dim=1)[1]
        # (bs, k)
        idx_k = idx[:, :k]
        gt_scores = torch.zeros((bs, k, num_points), device=sampling_scores.device)
        # (bs, k) -> (bs, k, np)
        for o in range(bs):
            for i in range(k):
                gt_scores[o, i, idx_k[o][i]] = 1
        loss = torch.sum(torch.mul(sampling_scores.log(), gt_scores),dim=(2, 1))
        loss = - loss / torch.sum(gt_scores, dim=(2, 1))
        loss = torch.mean(loss)
        return loss