#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#                    \file     losses.py
#                \author     chenghuige
#                    \date     2018-11-01 17:09:04.464856
#     \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import random


#https://github.com/xiangking/ark-nlp/blob/917c2f023ebbd6c80211b1eb3f30e6297213b070/ark_nlp/factory/loss_function/global_pointer_ce_loss.py#L5
class GlobalPointerCrossEntropy(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, CEFR_lvs=8, loss_weights=None):
        super(GlobalPointerCrossEntropy, self).__init__()

    @staticmethod
    def multilabel_categorical_crossentropy(y_true, y_pred, mask=None):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        loss = neg_loss + pos_loss
        if mask is not None:
            mask = mask.float()
            loss /= (mask.sum(-1) + 1e-12)
        return loss

    def forward(self, logits, target):
        """
                logits: [N, C, L, L]
        """
        bh = logits.shape[0] * logits.shape[1]
        target = torch.reshape(target, (bh, -1))
        logits = torch.reshape(logits, (bh, -1))
        # mask = (logits > -1e10).int()
        mask = None
        return torch.mean(
                GlobalPointerCrossEntropy.multilabel_categorical_crossentropy(
                        target, logits, mask))


# def pearson(pred, target):
#     cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#     score = cos(target - target.mean(dim=1, keepdim=True), pred - pred.mean(dim=1, keepdim=True))
#     return score

# def pearson(vx, vy):
#     return vx * vy * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))


def pearson(x, y):
    """
    Mimics `scipy.stats.pearsonr`
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
    Returns
    -------
    r_val : float
            pearsonr correlation coefficient between x and y
    Scipy docs ref:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    Scipy code ref:
            https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
            >>> x = np.random.randn(100)
            >>> y = np.random.randn(100)
            >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
            >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
            >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_den = torch.clamp(r_den, min=1e-6)
    r_val = r_num / r_den
    r_val = torch.clamp(r_val, min=-1., max=1.)
    return r_val


def pearson_loss(pred, target):
    return 1. - pearson(pred, target)


def spearman(pred, target):
    import torchsort
    x = 1e-3
    pred = torchsort.soft_rank(pred.reshape(1, -1), regularization_strength=x)
    target = torchsort.soft_rank(target.reshape(1, -1), regularization_strength=x)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    res = (pred * target).sum()
    res = torch.clamp(res, min=-1., max=1.)
    return res


def spearman_loss(pred, target):
    # return 1. - spearman(pred, target)
    from torchmetrics.functional import spearman_corrcoef
    return 1. - spearman_corrcoef(pred, target)


# https://github.com/YeYeYetta/CVPR2022-NAS-competition-Track-2-3rd-solution/blob/main/loss/Loss.ipynb
class TauLoss(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, pred, y, mask=None):
        pred = torch.transpose(pred, 0, 1)
        y = torch.transpose(y, 0, 1)
        if mask is None:
            return 1 - torch.cat([
                    self.get_score(pred[:, i], y[:, i]).reshape(1)
                    for i in range(y.shape[1])
            ]).reshape(1, -1).mean()
        else:
            # TODO not work..
            mask = torch.transpose(mask, 0, 1)
            return 1 - torch.cat([
                    self.get_score(pred[:mask[:, i].sum().item(), i],
                                                 y[:mask[:, i].item(), i]).reshape(1)
                    for i in range(y.shape[1])
            ]).reshape(1, -1).mean()

    def get_score(self, outputs, labels):
        output1 = outputs.unsqueeze(1).repeat(1, outputs.shape[0])
        label1 = labels.unsqueeze(1).repeat(1, labels.shape[0])

        tmp = ((output1 - output1.t()) * torch.sign(label1 - label1.t())).tanh()
        eye_tmp = tmp * torch.eye(tmp.shape[0]).cuda()
        new_tmp = tmp - eye_tmp

        loss = torch.sum(new_tmp) / (outputs.shape[0] * (outputs.shape[0] - 1))

        return loss


class RMSELoss(nn.Module):

    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class MCRMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.rmse = RMSELoss()
 
    def forward(self, y_pred, y_true):
        score = 0
        num_scored = y_true.shape[-1]
        for i in range(num_scored):
            score += self.rmse(y_pred[..., i], y_true[..., i]) / num_scored

        return score


class OLLoss10(nn.Module):

    def __init__(self, CEFR_lvs, pre_dist_mat=None, loss_weights=None, alpha=1.0):
        super().__init__()
        self.num_classes = CEFR_lvs
        self.loss_weights = loss_weights

        if pre_dist_mat is None:
            dist_mat = np.zeros((self.num_classes, self.num_classes))

            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    dist_mat[i][j] = np.abs(i-j)
        else:
            dist_mat = pre_dist_mat

        self.dist_mat = dist_mat
        self.alpha = alpha
 
    def forward(self, logits, labels):
        probas = torch.softmax(logits, dim=1)
        num_classes = self.num_classes
        dist_mat = self.dist_mat
        
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_mat[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances, device=logits.device, requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor**(self.alpha)
        if self.loss_weights is None:
            loss = torch.sum(err, axis=1).mean()
        else:
            loss = torch.sum(err, axis=1)
            weighted = self.loss_weights[labels]
            loss = (weighted * loss).mean()
        
        return loss


class OLLoss15(nn.Module):

    def __init__(self, CEFR_lvs, pre_dist_mat=None, loss_weights=None, alpha=1.5):
        super().__init__()
        self.num_classes = CEFR_lvs
        self.loss_weights = loss_weights

        if pre_dist_mat is None:
            dist_mat = np.zeros((self.num_classes, self.num_classes))

            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    dist_mat[i][j] = np.abs(i-j)
        else:
            dist_mat = pre_dist_mat

        self.dist_mat = dist_mat
        self.alpha = alpha
 
    def forward(self, logits, labels):
        probas = torch.softmax(logits, dim=1)
        num_classes = self.num_classes
        dist_mat = self.dist_mat
        
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_mat[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances, device=logits.device, requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor**(self.alpha)

        if self.loss_weights is None:
            loss = torch.sum(err, axis=1).mean()
        else:
            loss = torch.sum(err, axis=1)
            weighted = self.loss_weights[labels].to(device=logits.device)
            loss = (weighted * loss).mean()
        
        return loss

class CEB01OLLoss15(nn.Module):

    def __init__(self, CEFR_lvs, pre_dist_mat=None, loss_weights=None, alpha=1.5):
        super().__init__()
        self.num_classes = CEFR_lvs
        self.loss_weights = loss_weights
        
        if self.loss_weights:
            self.loss_ce = nn.CrossEntropyLoss(weight=self.loss_weights)
        else:
            self.loss_ce = nn.CrossEntropyLoss()

        if pre_dist_mat is None:
            dist_mat = np.zeros((self.num_classes, self.num_classes))

            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    dist_mat[i][j] = np.abs(i-j)
        else:
            dist_mat = pre_dist_mat

        self.dist_mat = dist_mat
        self.alpha = alpha
        self.beta = 0.1
 
    def forward(self, logits, labels):
        probas = torch.softmax(logits, dim=1)
        num_classes = self.num_classes
        dist_mat = self.dist_mat
        
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_mat[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances, device=logits.device, requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor**(self.alpha)

        if self.loss_weights is None:
            loss = torch.sum(err, axis=1).mean()
        else:
            loss = torch.sum(err, axis=1)
            weighted = self.loss_weights[labels].to(device=logits.device)
            loss = (weighted * loss).mean()

        ce_loss = self.loss_ce(logits, labels)
        loss = (1 - self.beta) * ce_loss + self.beta * loss
        return loss

class OLLoss20(nn.Module):

    def __init__(self, CEFR_lvs, pre_dist_mat=None, loss_weights=None, alpha=2.0):
        super().__init__()
        self.num_classes = CEFR_lvs
        self.loss_weights = loss_weights

        if pre_dist_mat is None:
            dist_mat = np.zeros((self.num_classes, self.num_classes))

            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    dist_mat[i][j] = np.abs(i-j)
        else:
            dist_mat = pre_dist_mat

        self.dist_mat = dist_mat
        self.alpha = alpha
 
    def forward(self, logits, labels):
        probas = torch.softmax(logits, dim=1)
        num_classes = self.num_classes
        dist_mat = self.dist_mat
        
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_mat[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances, device=logits.device, requires_grad=True)
        err = -torch.log(1-probas)*distances_tensor**(self.alpha)

        if self.loss_weights is None:
            loss = torch.sum(err, axis=1).mean()
        else:
            loss = torch.sum(err, axis=1)
            weighted = self.loss_weights[labels]
            loss = (weighted * loss).mean()
        
        return loss

class CORALLoss(nn.Module):

    def __init__(self, CEFR_lvs, weight=None):
        super().__init__()
        self.num_classes = CEFR_lvs
        # https://github.com/Raschka-research-group/coral-cnn/blob/726e54579db008d9c16868fa76b2292b9dec9fbc/model-code/cacd-coral.py#L117-L137
        self.loss_weights = weight[:self.num_classes -1]
 
    def forward(self, logits, labels, labels1):
        
        device = logits.device
        self.loss_weights = self.loss_weights.to(device)
        
        if self.loss_weights is None:
            err = (
                -torch.sum(
                    (
                        F.logsigmoid(logits)*labels + (F.logsigmoid(logits) - logits) * (1-labels)
                    ),
                    dim=1)
                )
        else:
            #self.loss_weights is not None:
            err = (
                -torch.sum(
                    (
                        F.logsigmoid(logits)*labels + (F.logsigmoid(logits) - logits) * (1-labels) 
                    ) * self.loss_weights,
                    dim=1)
                )
                
        loss = torch.mean(err)
        
        return loss
        
class CORNLoss(nn.Module):

    def __init__(self, CEFR_lvs):
        super().__init__()
        self.num_classes = CEFR_lvs
 
    def forward(self, logits, labels):
        
        device = logits.device
        num_classes = self.num_classes
        
        sets = []
        for i in range(num_classes-1):
            label_mask = labels > i-1
            label_tensor = (labels[label_mask] > i).to(torch.int64)
            sets.append((label_mask, label_tensor))

        num_examples = 0
        losses = 0.
        for task_index, s in enumerate(sets):
            train_examples = s[0]
            train_labels = s[1]

            if len(train_labels) < 1:
                continue

            num_examples += len(train_labels)
            pred = logits[train_examples, task_index]
            
            loss = -torch.sum(F.logsigmoid(pred)*train_labels
                                  + (F.logsigmoid(pred) - pred)*(1-train_labels)
                              )
            losses += loss
            
        return losses/num_examples


class OrdinalEntropy(nn.Module):
    def __init(self, lambda_d=0.5, lambda_t=0.5, margin=1.0):
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.margin = margin
    
    def forward(self, features, gt):
        """
        https://github.com/needylove/OrdinalEntropy/blob/main/DepthEstimation%26CrowdCounting/OrdinalEntropy.py#L8-L75
        Features: a certain layer's features
        gt: pixel-wise ground truth values, in depth estimation, gt.size()= n, h, w
        mask: In case values of some pixels do not exist. For depth estimation, there are some pixels lack the ground truth values
        """
        _gt = gt
        _features = features
        loss = 0

        """
        diverse part
        """
        u_value, u_index, u_counts = torch.unique(_gt, return_inverse=True, return_counts =True)
        center_f = torch.zeros([len(u_value), f_c]).cuda()
        center_f.index_add_(0, u_index, _features)
        u_counts = u_counts.unsqueeze(1)
        center_f = center_f / u_counts

        p = F.normalize(center_f, dim=1)
        _distance = euclidean_dist(p, p)
        _distance = up_triu(_distance)

        u_value = u_value.unsqueeze(1)
        _weight = euclidean_dist(u_value, u_value)
        _weight = up_triu(_weight)
        _max = torch.max(_weight)
        _min = torch.min(_weight)
        _weight = ((_weight - _min) / _max)

        _distance = _distance * _weight

        _entropy = torch.mean(_distance)
        loss = loss - self.lambda_d * _entropy

        """
        tightness part
        """
        _features = F.normalize(_features, dim=1)
        _features_center = p[u_index, :]
        _features = _features - _features_center
        _features = _features.pow(2)
        _tightness = torch.sum(_features, dim=1)
        _mask = _tightness > 0
        _tightness = _tightness[_mask]

        _tightness = torch.mean(_tightness)

        loss = loss + self.lambda_t * _tightness

        return loss
    
    def euclidean_dist(self, x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist
    
    
    def up_triu(self, x):
        # return a flattened view of up triangular elements of a square matrix
        n, m = x.shape
        assert n == m
        _tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
        return x[_tmp]

