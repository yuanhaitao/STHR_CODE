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
from commonLayers import Encoder, Decoder
import time
from distribution import TruncatedNormal

# dgl.batch
# DGLGraph
# Distpatcher是一个graph based的

EdgeGraph = namedtuple(
    'EdgeGraph', ['nodes', 'edges', 'out_list', 'in_list', 'node2edge'])


class NLLLoss(nn.Module):
    def __init__(self, speed_bins, alpha_1=0.5, alpha_2=0.5, lamb=0.5, device="cpu"):
        super(NLLLoss, self).__init__()
        self.lamb = lamb
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.speed_bins = nn.Parameter(torch.FloatTensor(
            speed_bins).to(device), requires_grad=False)
        self.speed_delta = nn.Parameter(torch.FloatTensor(
            speed_bins).to(device)-torch.FloatTensor(
            [0]+speed_bins[:-1]).to(device), requires_grad=False)
        # self.nllloss = nn.NLLLoss(reduce=False)  # 先不要reduce
        self.mseloss = nn.MSELoss(reduction="none")
        self.mapeloss = MAPELoss(reduce=False)
        self.maeloss = nn.SmoothL1Loss(reduction="none")
        self.mse_loss = TestNLLLoss(
            speed_bins, "mse", device)
        self.mae_loss = TestNLLLoss(
            speed_bins, "mae", device)
        self.mape_loss = TestNLLLoss(
            speed_bins, "mape", device)

        # self.trunc_norm = TruncatedNormal()

    def forward(self, ins, target):
        '''input: (edge_class, edge_bias, lens, src_node_mask, turning_class, turning_bias, src_edge_mask)
                edge_class = (batch_size, edge_lens, edge_speed_dim)
                edge_bias = (batch_size, edge_lens, edge_speed_dim)
                lens = (batch_size, edge_lens)
                src_node_mask = (batch_size, edge_lens)

           target: (batch_edge_value,
                    batch_turning_value)
                    batch_edge_value: (batch_size, edge_lens)
        '''
        # edge_class, edge_bias, lens_, src_node_mask = ins
        edge_delta, edge_bias, lens_, src_node_mask = ins
        batch_edge_value = target
        # print(edge_delta, edge_bias)
        batch_size, edge_len, _ = edge_bias.shape
        edge_speed = torch.sum((self.speed_bins.unsqueeze(0).unsqueeze(1).repeat(
            batch_size, edge_len, 1) - edge_delta * self.speed_delta.unsqueeze(0).unsqueeze(1).repeat(
            batch_size, edge_len, 1)) * edge_bias, dim=-1)
        # print(edge_speed)
        edge_time = lens_/edge_speed * 6.0
        link_loss = torch.sum(self.maeloss(
            edge_time, batch_edge_value).masked_fill(
            src_node_mask == 0, 0))/torch.sum(src_node_mask)

        # self.alpha_1*torch.sum(self.mseloss(
        #     edge_time, batch_edge_value).masked_fill(
        #     src_node_mask == 0, 0))/torch.sum(src_node_mask) + (1-self.alpha_1)*torch.sum(self.mapeloss(
        #         edge_time, batch_edge_value).masked_fill(
        #         src_node_mask == 0, 0))/torch.sum(src_node_mask)

        # route_loss = self.alpha_2 * \
        #     self.mse_loss(ins, target) + (1-self.alpha_2) * \
        #     self.mape_loss(ins, target)
        route_loss = self.mae_loss(ins, target)

        return link_loss, route_loss, link_loss*self.lamb+(1-self.lamb)*route_loss


class MAPELoss(nn.Module):
    def __init__(self, reduce=True):
        super(MAPELoss, self).__init__()
        self.reduce = reduce

    def forward(self, y_pred, y_true):
        if self.reduce:
            return torch.mean(torch.abs(y_pred - y_true)/y_true)*100
        else:
            return torch.abs(y_pred - y_true)/y_true*100


class TestNLLLoss(nn.Module):
    def __init__(self, speed_bins, kind="mse", device="cpu"):
        super(TestNLLLoss, self).__init__()
        self.speed_bins = nn.Parameter(torch.FloatTensor(
            speed_bins).to(device), requires_grad=False)
        self.speed_delta = nn.Parameter(torch.FloatTensor(
            speed_bins).to(device)-torch.FloatTensor(
            [0]+speed_bins[:-1]).to(device), requires_grad=False)
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
        # edge_class, edge_bias, lens_, src_node_mask = ins
        edge_delta, edge_bias, lens_, src_node_mask = ins
        batch_edge_value = target

        batch_size, edge_len, _ = edge_bias.shape
        edge_speed = torch.sum((self.speed_bins.unsqueeze(0).unsqueeze(1).repeat(
            batch_size, edge_len, 1) - edge_delta * self.speed_delta.unsqueeze(0).unsqueeze(1).repeat(
            batch_size, edge_len, 1)) * edge_bias, dim=-1)

        edge_time = (lens_/edge_speed * 6.0).masked_fill(
            src_node_mask == 0, 0)

        time = edge_time.sum(dim=-1)

        target_time = batch_edge_value.masked_fill(
            src_node_mask == 0, 0).sum(dim=-1)
        loss_ = self.loss(time, target_time)
        # print(loss_)
        return loss_


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim,
                 num_heads,
                 edge_type_num,  # 确定边的种类
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=True,
                 cat=True,
                 node_feature_dim=None,
                 edge_feature_dim=None,
                 use_static_feature=False):
        super(GATLayer, self).__init__()

        n_in_dim = in_dim
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = n_in_dim, n_in_dim
        self._out_feats = out_dim
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc_n = nn.Linear(n_in_dim, out_dim*num_heads, bias=False)

        self._cat = cat
        self._edge_type_num = edge_type_num
        self._node_feature_dim = node_feature_dim
        self._edge_feature_dim = edge_feature_dim

        self.use_static_feature = use_static_feature

        if self.use_static_feature:

            self.attn_l = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, node_feature_dim, self._edge_type_num)))
            self.attn_r = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, node_feature_dim, self._edge_type_num)))
            self.attn_m = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, edge_feature_dim, self._edge_type_num)))
        else:
            self.attn_l = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, out_dim, self._edge_type_num)))
            self.attn_r = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, out_dim, self._edge_type_num)))
            self.attn_m = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, edge_feature_dim, self._edge_type_num)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_dim:
                self.res_fc_n = nn.Linear(
                    self._in_dst_feats, num_heads * out_dim, bias=False)
            else:
                self.res_fc_n = nn.Identity()

        else:
            self.register_buffer('res_fc_n', None)
        self.reset_parameters()
        self.activation = None if activation == 'none' else (
            nn.ReLU() if activation == 'relu' else nn.Sigmoid())

    def edge_attn_udf(self, edges):
        left = torch.bmm(
            edges.src['el'], edges.data['mask'].view(-1, self._edge_type_num, 1))
        right = torch.bmm(
            edges.dst['er'], edges.data['mask'].view(-1, self._edge_type_num, 1))
        return {'e': (left+right+edges.data['em'])/3.0}

    def forward(self, graph, feat=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            if feat is None:
                n_feat = nn.Parameter(
                    graph.ndata['raw_pro'], requires_grad=False)
            else:
                n_feat = feat

            h_src = h_dst = self.feat_drop(n_feat)

            feat_src = feat_dst = self.fc_n(h_src).view(
                -1, self._num_heads, self._out_feats)

            e_mask = graph.edata['mask'].view(-1, self._edge_type_num, 1)
            if self.use_static_feature:
                n_feat = graph.ndata['pro'].unsqueeze(1).repeat(
                    1, self._num_heads, 1)

                e_feat = graph.edata['pro'].unsqueeze(1).repeat(
                    1, self._num_heads, 1)

                el = (n_feat.unsqueeze(3).repeat(1, 1, 1, self._edge_type_num)
                      * self.attn_l).sum(dim=-2).view(-1, self._num_heads, self._edge_type_num)
                er = (n_feat.unsqueeze(3).repeat(
                    1, 1, 1, self._edge_type_num)*self.attn_r).sum(dim=-2).view(-1, self._num_heads, self._edge_type_num)
                em = (e_feat.unsqueeze(3).repeat(1, 1, 1, self._edge_type_num) *
                      self.attn_m).sum(dim=-2).view(-1, self._num_heads, self._edge_type_num)

            else:
                el = (feat_src.unsqueeze(3).repeat(1, 1, 1, self._edge_type_num) *
                      self.attn_l).sum(dim=-2).view(-1, self._num_heads, self._edge_type_num)
                er = (feat_dst.unsqueeze(3).repeat(1, 1, 1, self._edge_type_num) *
                      self.attn_r).sum(dim=-2).view(-1, self._num_heads, self._edge_type_num)
                em = (graph.edata['pro'].unsqueeze(1).repeat(
                    1, self._num_heads, 1).unsqueeze(3).repeat(1, 1, 1, self._edge_type_num) *
                    self.attn_m).sum(dim=-2).view(-1, self._num_heads, self._edge_type_num)
            em = torch.bmm(em, e_mask)

            graph.edata.update({'em': em})
            feat_src = feat_src.view(-1, self._num_heads, self._out_feats)

            # graph.edata.update({'ft': feat_edge})

            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})

            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            # graph.apply_edges(fn.u_add_e('el', 'em', 'elm'))
            # graph.apply_edges(fn.e_add_v('elm','er','e'))
            graph.apply_edges(self.edge_attn_udf)
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(dglnn.edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            # graph.apply_edges(self.edge_feat_udf)
            rst_n = graph.dstdata['ft']

            # residual
            if self.res_fc_n is not None:
                resval_n = self.res_fc_n(h_dst).view(
                    h_dst.shape[0], -1, self._out_feats)
                rst_n = rst_n + resval_n

            if self._cat:
                rst_n = rst_n.view(-1, self._num_heads *
                                   self._out_feats)  # 最后用于改变结果
            else:
                rst_n = rst_n.view(-1, self._num_heads,
                                   self._out_feats).mean(1)  # 不改维度
            # activation
            if self.activation:
                rst_n = self.activation(rst_n)

            return rst_n

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_n.weight, gain=gain)
        # print(self.attn_l.shape)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_m, gain=gain)
        if isinstance(self.res_fc_n, nn.Linear):
            nn.init.xavier_normal_(self.res_fc_n.weight, gain=gain)


class GAT(nn.Module):
    def __init__(self, in_dim, num_layers,
                 num_hidden,
                 out_dim,
                 heads,
                 edge_type_num,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 node_feature_dim=None,
                 edge_feature_dim=None,
                 use_static_feature=False
                 ):
        # self.edge_graph = args['edge_graph']  # type: EdgeGraph
        # 根据edge_graph
        super(GAT, self).__init__()

        if in_dim != num_hidden*heads[0]:
            self.res_fc_n = nn.Linear(
                in_dim, num_hidden*heads[0], bias=False)
        else:
            self.res_fc_n = nn.Identity()

        # self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.i_gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATLayer(
            num_hidden*heads[0], num_hidden, heads[0], edge_type_num,
            feat_drop, attn_drop, negative_slope, False, self.activation, node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))

        # self.i_gat_layers.append(GATLayer(
        #     num_hidden*heads[0], num_hidden, heads[0], edge_type_num,
        #     feat_drop, attn_drop, negative_slope, False, self.activation, node_feature_dim=node_feature_dim,
        #     edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            # print(l, num_layers, heads)
            self.gat_layers.append(GATLayer(
                num_hidden * heads[l-1], num_hidden, heads[l], edge_type_num,
                feat_drop, attn_drop, negative_slope, residual, self.activation, node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))
            # self.i_gat_layers.append(GATLayer(
            #     num_hidden * heads[l-1], num_hidden, heads[l], edge_type_num,
            #     feat_drop, attn_drop, negative_slope, residual, self.activation, node_feature_dim=node_feature_dim,
            #     edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))
        # output projection
        self.gat_layers.append(GATLayer(
            num_hidden * heads[-2], out_dim, heads[-1], edge_type_num,
            feat_drop, attn_drop, negative_slope, residual, None, cat=False, node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))
        # self.i_gat_layers.append(GATLayer(
        #     num_hidden * heads[-2], out_dim, heads[-1], edge_type_num,
        #     feat_drop, attn_drop, negative_slope, residual, None, cat=False, node_feature_dim=node_feature_dim,
        #     edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))

    def forward(self, g, inv_g, h):
        # h = inputs
        # h = None
        # print("ok1")
        h = self.res_fc_n(h)
        for l in range(self.num_layers):
            h_ = self.gat_layers[l](g, h)
            # ih_ = self.i_gat_layers[l](inv_g, h)
            h += h_  # (h_ + ih_)/2.0
            # print(l, h[0].shape, h[1].shape)
        # output projection
        h_ = self.gat_layers[-1](g, h)
        # ih_ = self.i_gat_layers[-1](inv_g, h)
        h += h_  # (h_ + ih_)/2.0
        return h

# class Patcher Cell 包含一个GAT+一个GRU
# '''GAT解决对于input的修改，'''


# transformer decoder

GATParam = namedtuple('GATParam', ['in_dim', 'num_layers', 'num_hidden', 'out_dim', 'heads', 'edge_type_num', 'activation',
                                   'feat_drop', 'attn_drop', 'negtive_slope', 'residual', 'node_feature_dim', 'edge_feature_dim', 'use_static_feature'])

GRUParam = namedtuple('GRUParam', ['input_size', 'hidden_size',
                                   'num_layers', 'bias', 'batch_first', 'dropout', 'bidirectional'])

GRUCellParam = namedtuple('GRUCellParam', ['input_size', 'hidden_size'])


class RoadNetwork:
    def __init__(self, args):
        # 路网的空间属性信息
        self.node2edge_dict = args['node2edge_dict']  # 节点tuple转换到边id
        self.edge2node_dict = args['edge2node_dict']
        self.input_dict = args['input_dict']  # 每个节点对应到他的入度节点
        self.output_dict = args['output_dict']  # 每个节点对应到他的出度节点
        self.node_dict = args['node_dict']  # 节点的属性信息
        # {288416374: [0.6651350455486061, 0.8263341778903948, 3, 3]} (node id: [经度比率，维度比率，入度，出度])
        self.edge_dict = args['edge_dict']  # 边的属性信息
        '''{0: [0.6651350455486061,0.8263341778903948,0.6628707608956127,0.827280970796094,
        1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0, #highway
        0,0, #oneway
        0,1,0,0, #lanes
        1,0, # bridges
        1, #junction
        0, # tunel
        0.020517, #length (km)
        0.2 # maxspeed (100km/h)
        ]}  '''

        # 两条边的id转到index（数值tensor的第一维度）
        self.turning_dict = args['turning_dict']
        self.edge_graph = self.buidEdgeGraph()  # 五元组的表现形式（点，边，点的入度，点的出度，点对到边的映射）
        self.turning_types = ['back', 'forth', 'left', 'right']

    def buildDglGraph(self):
        assert self.edge_graph is not None
        source_list = []
        target_list = []
        edge_pros = []
        edge_mask = []
        for k in range(len(self.edge_graph.edges)):
            v = self.edge_graph.edges[k]
            source_list.append(v[1][0])
            target_list.append(v[1][1])
            edge_pros.append(v[2])
            edge_mask.append([0 for i in range(v[0])]+[1] +
                             [0 for i in range(len(self.turning_types)-v[0]-1)])
            # edge_mintime.append(v[2][-2]/v[2][-1]*60.0)
        nodes_pro = []
        # node_mintime = []
        for i in range(len(self.edge_graph.nodes)):
            v = self.edge_graph.nodes[i]
            nodes_pro.append(v)
            # node_mintime.append(v[-2]/v[-1]*60.0)
        g = dgl.graph((torch.tensor(source_list), torch.tensor(
            target_list)), num_nodes=len(self.edge_dict))

        inv_g = dgl.graph((torch.tensor(target_list), torch.tensor(
            source_list)), num_nodes=len(self.edge_dict))
        # print(type(g), type(inv_g))

        g.ndata['raw_pro'] = torch.FloatTensor(nodes_pro)  # 赋予节点相关的属性
        inv_g.ndata['raw_pro'] = torch.FloatTensor(nodes_pro)

        g.edata['raw_pro'] = torch.FloatTensor(edge_pros)
        g.edata['mask'] = torch.FloatTensor(edge_mask)

        inv_g.edata['raw_pro'] = torch.FloatTensor(edge_pros)
        inv_g.edata['mask'] = torch.FloatTensor(edge_mask)
        # inv_g.edata['mintime'] = torch.FloatTensor(edge_mintime)

        # g.ndata['mintime'] = torch.FloatTensor(node_mintime)

        return g, inv_g

    # 构建dgl能够识别的异构图（主要是获得异构图, 建立边的映射关系）
    def buildHeteroGraph(self):
        assert self.edge_graph is not None
        graph_keys = [("link", ty, "link") for ty in self.turning_types]
        graph_data = {}
        edge_pro_dict = {}
        for i, key in enumerate(graph_keys):
            sources = []
            targets = []
            edge_pros = []
            for k in range(len(self.edge_graph.edges)):
                v = self.edge_graph.edges[k]
                if v[0] == i:
                    sources.append(v[1][0])
                    targets.append(v[1][1])
                    edge_pros.append(v[2])
            graph_data[key] = (torch.tensor(sources), torch.tensor(targets))
            edge_pro_dict[self.turning_types[i]] = edge_pros

        g = dgl.heterograph(graph_data)

        # 构建逆向的图
        inverse_graph_data = {}
        for k, v in graph_data.items():
            inverse_graph_data[k] = (v[1], v[0])

        inv_g = dgl.heterograph(inverse_graph_data)

        # 获取节点的属性
        nodes_pro = []
        for i in range(len(self.edge_graph.nodes)):
            nodes_pro.append(self.edge_graph.nodes[i])
        g.ndata['pro'] = torch.FloatTensor(nodes_pro)  # 赋予节点相关的属性
        inv_g.ndata['pro'] = torch.FloatTensor(nodes_pro)

        for k, v in edge_pro_dict.items():
            g.edges[k].data['pro'] = torch.FloatTensor(v)
            inv_g.edges[k].data['pro'] = torch.FloatTensor(v)

        return g, inv_g

    # 构建一个edge作为节点，edge之间的连接作为边的图(边的类型需要)

    def buidEdgeGraph(self):
        nodes = {}  # 所有边的id对应到属性
        for k, v in self.edge_dict.items():
            nodes[k] = v
        edges = {}  # 所有turning的id及其对应到的属性
        out_dict = {}  # 节点间的拓扑结构
        in_dict = {}
        node2edge_dict = {}
        for k, v in self.turning_dict.items():
            e1, e2 = k
            n1 = self.edge2node_dict[e1][1]
            n2 = self.edge2node_dict[e2][0]
            assert n1 == n2
            e1_pro = self.edge_dict[e1]
            e2_pro = self.edge_dict[e2]
            n_pro = self.node_dict[n1]
            # 计算两条表的cos角度，然后
            vector1 = np.array([e1_pro[2]-e1_pro[0], e1_pro[3]-e1_pro[1]])
            vector2 = np.array([e2_pro[2]-e2_pro[0], e2_pro[3]-e2_pro[1]])
            cos_sim = np.dot(vector1, vector2) / \
                (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
            #
            if cos_sim <= -1+1e-3:  # 掉头
                type_turning = 0
            elif cos_sim >= np.cos(np.pi*1/4):  # 转弯幅度少于45度，判断为直行
                type_turning = 1
            else:  # 左转或者右转（将第一条路段映射到横坐标的正值，然后第二条路的最后一个点的y值如果大于0就是左转，否则为右转）
                cos1 = vector1[0]/np.linalg.norm(vector1)
                sin1 = vector1[1]/np.linalg.norm(vector1)
                # vector2_ = [e2_pro[2]-e1_pro[0], e2_pro[3]-e1_pro[1]] #计算第三个点
                # vector2_ = np.array([cos1*vector2_[0]+sin1*vector2_[1],-sin1*vector2_[0] +cos1*vector2_[1]])  #第三个点在在给定的第一个点的条件下，知道是左转还是右转
                if (e2_pro[2]-e1_pro[0])*(-sin1) + (e2_pro[3]-e1_pro[1])*cos1 >= 0:  # 右转
                    type_turning = 2
                else:  # 左转
                    type_turning = 3
            edges[v] = (type_turning, (e1,
                                       e2), e1_pro+e2_pro+n_pro)
            # 节点出度统计
            out_dict[e1] = out_dict.get(e1, []) + [e2]
            # 节点入度统计
            in_dict[e2] = in_dict.get(e2, [])+[e1]
            # 节点tuple对应的边统计
            node2edge_dict[(e1, e2)] = v
        return EdgeGraph._make([nodes, edges, in_dict, out_dict, node2edge_dict])


class Predictor(nn.Module):
    def __init__(self, in_feat, out_feat, activation):
        self.predictor = nn.Linear(in_feat, out_feat)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.predictor(x))


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.lr = nn.Linear(10, 1)

    def forward(self, x):
        return self.lr(x)


class LocalGraphEncoder(nn.Module):
    def __init__(self, args):
        super(LocalGraphEncoder, self).__init__()
        self.device = args['device']
        self.gat_param = GATParam(**args['localgraph_gat_param'])
        self.use_gat = args['use_local_gat']
        if self.use_gat:
            self.gat = GAT(*tuple(self.gat_param))
            self.att = nn.Linear(
                self.gat_param.node_feature_dim, self.gat_param.out_dim)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.gat_param.in_dim, self.gat_param.num_hidden),
                nn.ReLU(),
                nn.Linear(self.gat_param.num_hidden, self.gat_param.out_dim)
            )

        print(self.gat_param)
        self.out_dim = self.gat_param.out_dim

    def forward(self, g, inv_g, x, idx, mask):
        # print(len(gs), len(inv_gs))
        # g = dgl.batch(gs)
        # inv_g = dgl.batch(inv_gs)
        node_x = x
        node_idx = idx
        node_mask = mask
        _, lens, _ = node_x.shape
        node_res = []
        if self.use_gat:
            # print(node_x.shape, edge_x.shape)
            for i in range(lens):
                node_h = self.gat(
                    g, inv_g, node_x[:, i, :])
                # 将节点和边的表示进行平均融合
                node_res.append(node_h.unsqueeze(1))

            node_res = torch.cat(node_res, dim=1)
            bs, max_node_len = node_idx.shape

            node_res = node_res.index_select(
                index=node_idx.view(-1), dim=0).view(bs, max_node_len, lens, -1)

            node_features = self.att(g.ndata['pro']).index_select(
                index=node_idx.view(-1), dim=0).view(bs, max_node_len, self.out_dim).unsqueeze(2).repeat(1, 1, lens, 1)

            # print(node_res.shape, node_features.shape, node_mask.shape)
            score = torch.sum(node_res*node_features, dim=-1).masked_fill(
                node_mask.unsqueeze(2).repeat(1, 1, lens) == 0, 1e-20)

            score = torch.softmax(score, dim=1).unsqueeze(3)
            node_mean = torch.sum(node_res * score, dim=1)
        # node_mean = torch.sum(node_res.index_select(index=node_idx.view(-1), dim=0).view(bs, max_node_len, lens, -1)
        #                       * node_mask.unsqueeze(2).unsqueeze(3), dim=1) / torch.sum(node_mask, dim=1).unsqueeze(1).unsqueeze(2)
        else:
            node_res = self.encoder(node_x)
            node_mean = torch.mean(node_res, dim=1)
        return node_mean


class LeimPatcher(nn.Module):
    def __init__(self, args):
        super(LeimPatcher, self).__init__()
        # out_dim = args['out_dim']

        #
        self.use_gat = args['use_global_gat']
        self.n_split_dim = args['n_split_dim']
        self.device = args['device']

        self.g, self.inv_g = args['g'], args['inv_g']
        if args['road_network'] is not None:
            self.node_pro_dim = len(
                args['road_network'].node_dict[0])  # 原始node
            self.edge_pro_dim = len(
                args['road_network'].edge_dict[0])  # 原始edge
            self.node_size = len(args['road_network'].edge_graph.nodes)
            self.edge_size = len(args['road_network'].edge_graph.edges)
        else:
            self.node_pro_dim = self.edge_pro_dim = self.node_size = self.edge_size = 10

        # 公用的参数
        dropout = args['dropout']
        # 路段、转化点的隐向量维度
        edge_hidden_size = args['edge_hidden_size']
        turning_hidden_size = args['turning_hidden_size']

        self.patcher_gat_param = GATParam(**args['patcher_gat_param'])
        print(self.patcher_gat_param)

        # 道路属性的编码（meta-learner）
        print("edge_pro_dim is: {}".format(self.edge_pro_dim))
        self.edge_weight_nn = nn.Sequential(
            nn.Linear(self.edge_pro_dim, edge_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_size,
                      self.patcher_gat_param.node_feature_dim),
            # nn.Sigmoid()
        )
        # 道路交叉点使用两条边加中间节点的属性信息（meta-learner）
        self.turning_weight_nn = nn.Sequential(
            nn.Linear(
                2*self.edge_pro_dim+self.node_pro_dim, turning_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(turning_hidden_size,
                      self.patcher_gat_param.edge_feature_dim),
            # nn.Sigmoid()
        )

        if self.use_gat:
            # 构建patcher
            self.patcher_gat = GAT(*tuple(self.patcher_gat_param))
        else:
            self.patcher_gat = nn.Sequential(
                nn.Linear(self.patcher_gat_param.in_dim,
                          self.patcher_gat_param.num_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.patcher_gat_param.num_hidden,
                          self.patcher_gat_param.out_dim)
            )

    def to(self, device):
        super().to(device)
        self.g, self.inv_g = self.g.to(
            device), self.inv_g.to(device)

    def forward(self, n_all):
        # is_train: 如果是训练的话，那么y是和x具有一样的表示，都是稀疏的分布表示
        self.g.ndata['pro'] = self.edge_weight_nn(self.g.ndata['raw_pro'])
        self.inv_g.ndata['pro'] = self.edge_weight_nn(
            self.inv_g.ndata['raw_pro'])
        self.g.edata['pro'] = self.turning_weight_nn(self.g.edata['raw_pro'])
        self.inv_g.edata['pro'] = self.turning_weight_nn(
            self.inv_g.edata['raw_pro'])
        if self.use_gat:
            node_h = self.patcher_gat(self.g, self.inv_g, n_all)
        else:
            node_h = self.patcher_gat(n_all)
        # 过一下GRU
        # node_h, edge_h = self.gruprocess(self.g, (node_h, edge_h), None)
        # node_h = n_all
        return node_h, self.g.ndata['raw_pro'][:, -2]


'''实现一个时空融合的transformer编码和解码器得到每条路段以及每个交叉口的通勤时间'''

STParam = namedtuple('STParam', ['embedding_size_1', 'lens', 'embedding_size_2', 'hidden_size', 'num_layers', 'num_heads', 'total_key_depth', 'total_value_depth',
                                 'filter_size', 'max_length', 'input_dropout', 'layer_dropout',
                                 'attention_dropout', 'relu_dropout', 'use_mask', 'act', 'day_embedding_dim', 'time_embedding_dim', 'slot_size', 'edge_speed_dim'])

# class STTransformer(nn.Module):
#     def __init__(self, args):


class STTransformer(nn.Module):
    """
    A Transformer Module For BabI data.
    Inputs should be in the shape story: [batch_size, memory_size, story_len ]
                                  query: [batch_size, 1, story_len]
    Outputs will have the shape [batch_size, ]
    """

    def __init__(self, embedding_size_1, lens, embedding_size_2, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length, input_dropout, layer_dropout,
                 attention_dropout, relu_dropout, use_mask, act, day_embedding_dim, time_embedding_dim, slot_size, edge_speed_dim, use_encoder_attention, use_decoder_attention):
        super(STTransformer, self).__init__()
        self.use_e_att = use_encoder_attention
        self.use_d_att = use_decoder_attention
        # self.encoder_input_size = embedding_size #
        self.transformer_enc = Encoder(embedding_size_1, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                                       filter_size,  lens, input_dropout, layer_dropout,
                                       attention_dropout, relu_dropout, act, use_att=self.use_e_att)

        self.transformer_dec = Decoder(embedding_size_2, hidden_size, num_layers, num_heads, total_key_depth, total_key_depth, filter_size,
                                       max_length, input_dropout, layer_dropout, attention_dropout, relu_dropout, act, use_mask=use_mask, use_att=self.use_d_att)

        self.merge_each_step = nn.Linear(
            day_embedding_dim+hidden_size+time_embedding_dim, hidden_size)
        self.edge_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(relu_dropout),
            nn.Linear(hidden_size, edge_speed_dim)
        )

        # 需要一个EDM转移矩阵，该矩阵将原来的dist转换成新的dist
        self.edge_dist_transformer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(relu_dropout),
            nn.Linear(hidden_size, edge_speed_dim*edge_speed_dim)
        )

        # self.edge_bias = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(relu_dropout),
        #     nn.Linear(hidden_size, edge_speed_dim)
        # )

        self.edge_speed_dim = edge_speed_dim

    def forward(self, x_enc, x_dec, src_mask, k_, q_, n_):
        # x_enc: [batch_size, enc_seq_length, w, h, 2]
        # x_dec: [batch_size, dec_seq_length, node_edge_embedding]
        # src_mask: [batch_size, dec_seq_length]
        # n_: [batch_size, dec_node_length]
        # e_: [batch_size, dec_edge_length]
        # k_: [batch_size, enc_seq_length, day_dim+time_dim]

        # if torch.isnan(x_enc).any():
        #     print(x_enc)
        #     assert torch.isnan(x_enc).any() == False

        encoder_output, _ = self.transformer_enc(x_enc)
        # encoder_output通过时间的embedding得到scale
        # print("d_t.shape", d_t[0].shape)

        # print("scale_.shape", q_.shape, k_.shape, encoder_output.shape)
        scale_ = torch.bmm(k_, q_.unsqueeze(2))  # .unsqueeze(-1).unsqueeze(-1)

        # print("encoder_output.shape", encoder_output.shape, scale_.shape)
        encoder_output = encoder_output + encoder_output*scale_
        # encoder_output = self.merge_each_step(
        #     torch.cat([encoder_output, k_], dim=-1)) + encoder_output

        decoder_output, _ = self.transformer_dec(
            x_dec, encoder_output, src_mask)

        src_node_mask = src_mask

        # edge_bias = n_  # torch.softmax(self.edge_bias(decoder_output), dim=-1)

        edge_bias = torch.softmax(self.edge_dist_transformer(decoder_output).view(
            *tuple(decoder_output.shape[:-1]), self.edge_speed_dim, self.edge_speed_dim), dim=-2)
        edge_bias = torch.matmul(edge_bias, n_.view(
            *tuple(n_.shape), 1)).squeeze(dim=-1)
        # mu_ = n_
        edge_delta = torch.sigmoid(
            self.edge_generator(decoder_output))
        # edge_delta = torch.sigmoid(torch.tanh(
        #     self.edge_generator(decoder_output))*std_ + mu_)
        return edge_delta, edge_bias, src_node_mask

# encoder 每一个抽取出来的local图


class Leim(nn.Module):
    def __init__(self, args):
        super(Leim, self).__init__()
        # self._pather = LeimPatcher(args)  # 获得所有的link对应的时间信息，然后需要根据具体的link表示抽取出特定的

        self.edge_speed_dim = args['n_split_dim']
        day_embedding_dim = args['stparam']['day_embedding_dim']
        time_embedding_dim = args['stparam']['time_embedding_dim']
        slot_size = args['stparam']['slot_size']
        # gat的时候的输入也是embedding_size_2
        args['patcher_gat_param']['in_dim'] = args['link_embedding_dim']

        self._pather = LeimPatcher(args)

        # args['localgraph_gat_param']['in_dim'] = 2 + \
        #     len(
        #         args['road_network'].edge_dict[0])
        self._localgraph = LocalGraphEncoder(args)

        args['stparam']['edge_speed_dim'] = args['n_split_dim']
        # self.edge_speed_dim  # * 2
        args['stparam']['embedding_size_2'] = self.edge_speed_dim

        # args["patcher_gat_param"]["out_dim"] + \
        #     day_embedding_dim+time_embedding_dim
        stParam = STParam(**args['stparam'])
        self._transformer = STTransformer(
            *tuple(stParam), use_encoder_attention=args['use_encoder_attention'], use_decoder_attention=args['use_decoder_attention'])

        # self.nodes_ = nn.Parameter(torch.LongTensor(
        #     [i for i in range(args['node_size']+1)]), requires_grad=False)

        # 将edge和node映射到同一维度上去embedding_size_2是空间点和边的编码表示
        # 道路属性的编码（meta-learner）
        # self.node_emb = nn.Embedding(
        #     args['node_size']+1, args['link_embedding_dim'], padding_idx=args['node_size'])
        self.node_emb = nn.Sequential(
            nn.Linear(self._pather.edge_pro_dim, args['link_embedding_dim']),
            nn.ReLU(),
            nn.Linear(args['link_embedding_dim'],
                      args['link_embedding_dim']
                      )
        )
        # 时间编码
        self.day_embedding = nn.Embedding(8, day_embedding_dim)
        self.time_embedding = nn.Embedding(
            int(24*60/slot_size)+1, time_embedding_dim)

        self.node_size = args['node_size']
        self.edge_size = args['edge_size']
        self.turning_mask = nn.Parameter(
            args['g'].edata['mask'], requires_grad=False)
        self.inv_turning_mask = nn.Parameter(
            args['inv_g'].edata['mask'], requires_grad=False)

        # self.base_n_proj = nn.Sequential(
        #     nn.Linear(
        #         args["patcher_gat_param"]["out_dim"]+day_embedding_dim+time_embedding_dim, args["patcher_gat_param"]["out_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["patcher_gat_param"]["out_dim"], args["patcher_gat_param"]
        #               ["out_dim"]+day_embedding_dim+time_embedding_dim))
        # 周期性质和节点特征做矩阵乘法得到基准预测结果
        self.base_n_mu = nn.Sequential(
            nn.Linear(
                args["patcher_gat_param"]["out_dim"]+day_embedding_dim+time_embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(args['dropout']),
            nn.Linear(16, self.edge_speed_dim))
        # nn.Linear(
        #     args["patcher_gat_param"]["out_dim"]+day_embedding_dim+time_embedding_dim, self.edge_speed_dim)
        # self.base_n_sigma = nn.Linear(
        #     args["patcher_gat_param"]["out_dim"]+day_embedding_dim+time_embedding_dim, self.edge_speed_dim)
        # self.base_n_delta = nn.Linear(
        #     args["patcher_gat_param"]["out_dim"]+day_embedding_dim+time_embedding_dim, self.edge_speed_dim)

    def forward(self, x):
        d_t, x, src_mask = x
        # x_enc:
        # node_dec: [batch_size, dec_node_length]
        # edge_dec: [batch_size, dec_edge_length]
        # src_mask: [bath_size, dec_node_length+dec_edge_length]
        # start_time = time.time()
        n_all = self.node_emb(
            self._pather.g.ndata['raw_pro'])  # 将所有的都拿出来
        node_h, lens_ = self._pather(n_all)
        # 基准
        # n_ = torch.softmax(torch.matmul(
        #     node_h, self.base_n_mu).squeeze(), dim=-1

        # print("generate n_, e_, used time is: {}".format(time.time()-start_time))
        x_enc, node_dec = x

        # 获得encoder的表示
        batch_enc_turning, batch_enc_edges, node_lens, batch_enc_src, batch_enc_dst, x, idxs, masks = x_enc

        sub_g = dgl.graph((batch_enc_src, batch_enc_dst),
                          num_nodes=sum(node_lens))
        sub_inv_g = dgl.graph(
            (batch_enc_dst, batch_enc_src), num_nodes=sum(node_lens))
        sub_g.edata['mask'] = self.turning_mask.index_select(
            0, batch_enc_turning)
        sub_inv_g.edata['mask'] = self.turning_mask.index_select(
            0, batch_enc_turning)
        sub_g.edata['pro'] = self._pather.g.edata['pro'].index_select(
            0, batch_enc_turning)
        sub_inv_g.edata['pro'] = self._pather.inv_g.edata['pro'].index_select(
            0, batch_enc_turning)
        sub_g.ndata['pro'] = self._pather.g.ndata['pro'].index_select(
            0, batch_enc_edges)
        sub_inv_g.ndata['pro'] = self._pather.inv_g.ndata['pro'].index_select(
            0, batch_enc_edges)

        # x = torch.cat([x, self._pather.g.ndata['raw_pro'].index_select(0, batch_enc_edges).unsqueeze(
        #     1).repeat(1, x.shape[1], 1)], dim=-1)
        # x = torch.zeros_like(x).to(x.device)
        x_enc = self._localgraph(sub_g, sub_inv_g, x, idxs, masks)

        bs, node_len = node_dec.shape
        n_idx = node_dec.masked_fill(
            node_dec == self.node_size, 0)

        day_embedding = self.day_embedding(d_t[0])
        time_embedding = self.time_embedding(d_t[1])
        q_ = torch.cat(
            [day_embedding[:, -1, :], time_embedding[:, -1, :]], dim=-1)

        k_ = torch.cat([day_embedding[:, :-1, :],
                        time_embedding[:, :-1, :]], dim=-1)

        node_dec = node_h.index_select(
            index=n_idx.view(-1), dim=0).view(bs, node_len, -1)  # self.

        node_dec = torch.cat(
            [node_dec, q_.unsqueeze(1).repeat(1, node_len, 1)], dim=-1)

        # x_dec = node_dec
        # print(x_dec.shape)
        # node_dec = self.base_n_proj(node_dec) + node_dec
        # sigma_ = torch.relu(self.base_n_sigma(
        #     node_dec))

        # [mu_, sigma_]  # , self.base_n_delta(node_dec))
        # x_dec = self.base_n_mu(node_dec)
        n_ = torch.softmax(self.base_n_mu(node_dec), dim=-1)
        x_dec = n_

        # spatial_ = n_all.index_select(
        #     index=n_idx.view(-1), dim=0).view(bs, node_len, -1)
        # x_dec = self.base_n_mu(node_dec)
        # torch.cat(
        # [node_dec, q_.unsqueeze(1).repeat(1, node_len, 1)], dim=-1)

        # x_dec = torch.cat([mu_, ], dim=-1)

        lens_ = lens_.index_select(
            index=n_idx.view(-1), dim=0).view(bs, node_len)

        edge_delta, edge_bias, src_node_mask = self._transformer(
            x_enc, x_dec, src_mask, k_, q_, n_)
        # print("generate transformer, used time is: {}".format(
        #     time.time()-start_time))
        return edge_delta, edge_bias, lens_, src_node_mask

    def to(self, device):
        super().to(device)
        self._pather.to(device)
