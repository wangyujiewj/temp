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
    def forward(self, sampling_scores, gt_scores):
        loss = torch.sum(torch.mul(sampling_scores.log(), gt_scores), dim=(2, 1))
        loss = - loss / torch.sum(gt_scores, dim=(2, 1))
        loss = torch.mean(loss)
        return loss

class MCCLossFuncv3(nn.Module):
    def __init__(self):
        super(MCCLossFuncv3, self).__init__()
        return
    def exp_mcc(self, loss_, beta, sigma_, batch_size):
        exp_error = torch.exp(-(loss_ * 0.5 * sigma_ ** (-2)))
        one_tensor = exp_error.new_tensor(1).expand(batch_size, )
        mcc_loss = - exp_error + one_tensor
        mcc_loss = mcc_loss * beta.squeeze(-1)
        mean_mcc_loss = torch.mean(mcc_loss, dim=0)
        return mean_mcc_loss
    def forward(self, R_pred, t_pred, R_gt, t_gt, sigma_):
        batch_size = R_pred.size(0)
        identity = torch.eye(3, device=R_gt.device).unsqueeze(0).repeat(batch_size, 1, 1)
        # sigma_ = sigma_.view(batch_size, 1)
        # (bs,)
        beta = (1 - torch.exp(-(0.5 * sigma_**(-2)))) ** (-1)
        # (bs,)
        r_loss = torch.mean((torch.matmul(R_pred.transpose(2, 1), R_gt) - identity) ** 2, dim=(2, 1))
        t_loss = torch.mean((t_pred - t_gt) ** 2, dim=-1)
        r_mcc_loss = self.exp_mcc(r_loss, beta, sigma_, batch_size)
        t_mcc_loss = self.exp_mcc(t_loss, beta, sigma_, batch_size)
        return r_mcc_loss + t_mcc_loss