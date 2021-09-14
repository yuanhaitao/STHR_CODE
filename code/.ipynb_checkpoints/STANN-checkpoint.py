# -*- encoding: utf-8 -*-
import copy
from dgl._deprecate.graph import DGLGraph, batch
from dgl._ffi.function import get_global_func
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import numpy as np
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from collections import namedtuple
import torch.nn.functional as F
import copy
# from torch.nn import LayerNorm
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
import time
from dgl.nn import GATConv


class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true)/y_true)*100


class BasicLoss(nn.Module):
    def __init__(self, kind="mse", device="cpu"):
        super(BasicLoss, self).__init__()
        if kind == 'mse':
            self.loss = nn.MSELoss()
        elif kind == 'mae':
            self.loss = nn.SmoothL1Loss()
        elif kind == 'mape':
            self.loss = MAPELoss()
        else:
            print("loss kind should be figured as one of mse, mae and mape")
            assert 1 == 0

    def forward(self, ins, target):
        '''input: (edge_class, edge_bias, lens, src_node_mask, turning_class, turning_bias, src_edge_mask)
                edge_class = (batch_size, edge_lens, edge_speed_dim)
                edge_bias = (batch_size, edge_lens, 1)
                lens = (batch_size, edge_lens)
                src_node_mask = (batch_size, edge_lens)
                turning_class = (batch_size, turning_lens, turning_time_dim)
                turning_bias = (batch_size, turning_lens, 1)

           target: (batch_edge_value,
                    batch_turning_value)
                    batch_edge_value: (batch_size, edge_lens)
                    batch_turning_value: (batch_size, turning_lens)
        '''
        H_cells, src_node_mask = ins
        batch_edge_value = target

        time = H_cells.masked_fill(src_node_mask == 0, 0).sum(dim=-1)

        target_time = batch_edge_value.masked_fill(
            src_node_mask == 0, 0).sum(dim=-1)
        loss_ = self.loss(time, target_time)
        return loss_


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        # print(query.shape, key.shape, value.shape, mask.shape)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, heads, q_dim, k_dim, v_dim, d_hidden, d_out, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        # We assume d_v always equals d_k
        self.heads = heads
        self.d_hidden = d_hidden
        self.query_dense = nn.Linear(q_dim, d_hidden*heads)
        self.key_dense = nn.Linear(k_dim, d_hidden*heads)
        self.value_dense = nn.Linear(v_dim, d_hidden*heads)

        self.output_linear = nn.Linear(self.heads*d_hidden, d_out)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = self.query_dense(query).view(
            batch_size, -1, self.heads, self.d_hidden).transpose(1, 2).contiguous(), self.key_dense(key).view(
            batch_size, -1, self.heads, self.d_hidden).transpose(1, 2).contiguous(), self.value_dense(value).view(
            batch_size, -1, self.heads, self.d_hidden).transpose(1, 2).contiguous()
        if mask != None:
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.heads * self.d_hidden)

        return self.output_linear(x)


class STANN(nn.Module):
    def __init__(self, args):
        super(STANN, self).__init__()

        g = dgl.graph((args['src'], args['dst']),
                      num_nodes=args['link_num'])
        self.g = dgl.add_self_loop(g)
        self.edge_features = nn.Parameter(
            torch.FloatTensor(args['edge_features']), requires_grad=False)  # 保存特征信

        self.spatial_embedding_dim = args['spatial_embedding_dim']

        self.gatConv = GATConv(
            len(args['edge_features'][0]), self.spatial_embedding_dim, num_heads=1)

        # 使用lstm
        self.lstm = nn.LSTM(
            self.spatial_embedding_dim, args['lstm_hidden'], batch_first=True)

        # 使用Attention
        self.attention = MultiHeadedAttention(args['heads'], args['lstm_hidden'], args['lstm_hidden'],
                                              args['lstm_hidden'], args['d_hidden'], args['d_out'], args['dropout'])

        self.output = nn.Sequential(
            nn.Linear(args['d_out'], 1)
        )

    def forward(self, x):
        (_, _), (src_idx,
                 src_mask), (_, x_idx, x_mask, _) = x
        # batch_day: [batch_size, pre_slot_num+1]
        # batch_time: [batch_size, pre_slot_num+1]
        # src_idx: [batch_size, max_seq]
        # src_mask: [batch_size, max_seq]
        # x_data: [all_link_num, max_neighbors_num, pre_slot_num, traffic_dim]
        # x_idx: [all_link_num, max_neighbors_num+1] #第一个元素是该link对应的id
        # x_mask: [all_link_num, max_neighbors_num]
        # x_flag: [all_link_num]

        batch_size, max_seq = src_idx.shape
        all_link_num, max_neighbors_num = x_mask.shape
        assert max_neighbors_num+1 == x_idx.shape[1]

        gat_res = self.gatConv(
            self.g, self.edge_features).view(-1, self.spatial_embedding_dim)

        data = gat_res.index_select(
            dim=0, index=x_idx[:, 0].view(-1)).view(all_link_num, -1)

        # 从data中抽取出batch_size的形状
        data = data.index_select(
            dim=0, index=src_idx.view(-1)).view(batch_size, max_seq, -1)
        # 开始使用lstm
        output, _ = self.lstm(data)
        # batch, max_seq, num_directions * hidden_size
        output = self.attention(output, output, output, mask=src_mask.unsqueeze(
            1).repeat(1, max_seq, 1)*src_mask.unsqueeze(2).repeat(1, 1, max_seq))
        output = self.output(output).squeeze()
        return output, src_mask

    def to(self, device):
        super().to(device)
        self.g = self.g.to(
            device)
