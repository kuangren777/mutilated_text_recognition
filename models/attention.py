# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 23:41
# @Author  : KuangRen777
# @File    : attention.py
# @Tags    :
import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        q = self.W(x)
        k = self.W(x)
        v = self.W(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.W.in_features ** 0.5)
        attn_dists = torch.nn.functional.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_dists, v)
        return attn_output
