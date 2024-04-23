#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math
import numpy as np

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="diou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):  # (xi.yi,w,h)
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-7)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2  ###############   2>>>>3(a-iou)
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-7)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)  ####################**2

        elif self.loss_type == "diou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)  # 包围框的左上点
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)  # 包围框的右下点
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            diou = iou - (center_dis / convex_dis)
            loss = 1 - diou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == "ciou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            v = (4 / math.pi ** 2) * torch.pow(torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-7)) -
                                               torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-7)), 2)

            with torch.no_grad():
                alpha = v / ((1 + 1e-7) - iou + v)

            ciou = iou - (center_dis / convex_dis + alpha * v)

            loss = 1 - ciou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == "siou":

            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )

            s_cw = (c_br - c_tl)[:, 0]
            s_ch = (c_br - c_tl)[:, 1]
            cw = target[:, 0] - pred[:, 0]
            ch = target[:, 1] - pred[:, 1]
            sigma = torch.pow(cw ** 2 + ch ** 2, 0.5)
            sin_alpha = torch.abs(ch) / sigma
            sin_beta = torch.abs(cw) / sigma
            thres = torch.pow(torch.tensor(2.), 0.5) / 2
            sin_alpha = torch.where(sin_alpha < thres, sin_alpha, sin_beta)
            angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha)-np.pi / 4), 2)

            gamma = angle_cost - 2
            rho_x = (cw / s_cw) ** 2
            rho_y = (ch / s_ch) ** 2
            delta_x = 1 - torch.exp(gamma * rho_x)
            delta_y = 1 - torch.exp(gamma * rho_y)
            distance_cost = delta_x + delta_y

            w_gt = target[:, 2]
            h_gt = target[:, 3]
            w_pred = pred[:, 2]
            h_pred = pred[:, 3]
            W_w = torch.abs(w_pred - w_gt) / torch.max(w_pred, w_gt)
            W_h = torch.abs(h_pred - h_gt) / torch.max(h_pred, h_gt)

            theta = 4
            shape_cost = torch.pow((1 - torch.exp(-1 * W_w)), theta) + torch.pow((1 - torch.exp(-1 * W_h)), theta)
            siou = iou - (distance_cost + shape_cost) * 0.5

            loss = 1 - siou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == "eiou":

            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            dis_w = torch.pow(pred[:, 2] - target[:, 2], 2)  # 两个框的w欧式距离
            dis_h = torch.pow(pred[:, 3] - target[:, 3], 2)  # 两个框的h欧式距离

            C_w = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + 1e-7  # 包围框的w平方
            C_h = torch.pow(c_br[:, 1] - c_tl[:, 1], 2) + 1e-7  # 包围框的h平方

           #加
            v = dis_w / C_w + dis_h / C_h
            alpha = v / ((1 + 1e-7) - iou + v)
            #eiou = iou - (center_dis / convex_dis) - alpha * v
            eiou = iou - (center_dis / convex_dis + dis_w / C_w + dis_h / C_h)
            loss = 1 - eiou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class alpha_IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="ciou", alpha=3):
        super(alpha_IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

        self.alpha = alpha

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-7)

        if self.loss_type == "iou":
            loss = 1 - iou ** self.alpha  ###############   2>>>>3(a-iou)
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou ** self.alpha - ((area_c - area_u) / area_c.clamp(1e-16)) ** self.alpha
            loss = 1 - giou.clamp(min=-1.0, max=1.0)


        elif self.loss_type == "ciou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            v = (4 / math.pi ** 2) * torch.pow(torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-7)) -
                                               torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-7)), 2)
            with torch.no_grad():
                beat = v / (v - iou + 1)

            ciou = iou ** self.alpha - (center_dis ** self.alpha / convex_dis ** self.alpha + (beat * v) ** self.alpha)

            loss = 1 - ciou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
