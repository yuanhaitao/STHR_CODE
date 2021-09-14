# -*- encoding: utf-8 -*-
import copy
from dgl._deprecate.graph import DGLGraph
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

# dgl.batch
# DGLGraph
# Distpatcher是一个graph based的

EdgeGraph = namedtuple(
    'EdgeGraph', ['nodes', 'edges', 'out_list', 'in_list', 'node2edge'])


class NLLLoss(nn.Module):
    def __init__(self, eps=1, lamb=0.5, use_weight=False):
        super(NLLLoss, self).__init__()
        self.eps = eps
        self.lamb = lamb
        self.use_weight = use_weight

    def forward(self, ins, target):
        '''input: (n_mu, n_sigma, e_mu, e_sigma)
           target: (batch_node_key, batch_node_value, batch_edge_key, batch_edge_value)
                batch_node_key = (N): flatten_index
                batch_node_value = (N)
                batch_edge_key = (M ): flatten_index
                batch_edge_key = (M)
        '''
        n_mu, n_sigma2, e_mu, e_sigma2 = ins
        batch_node_key, batch_node_value, batch_edge_key, batch_edge_value = target
        pre_node_mu = n_mu.take(batch_node_key)
        pre_node_sigma2 = n_sigma2.take(batch_node_key)
        pre_edge_mu = e_mu.take(batch_edge_key)
        pre_edge_sigma2 = e_sigma2.take(batch_edge_key)

        # print(batch_node_value.shape)

        # print(batch_node_value, batch_edge_value)
        # print(torch.sum(n_mu < 0), torch.sum(n_sigma2 < 0),
        #       torch.sum(e_mu < 0), torch.sum(e_sigma2 < 0))
        # print(torch.sum(batch_node_value < 0),
        #       torch.sum(batch_edge_value < 0))
        # print(torch.mean((batch_node_value-pre_node_mu)
        #                  ** 2/(self.eps+pre_node_sigma2)), torch.mean((batch_edge_value-pre_edge_mu)**2/(self.eps+pre_edge_sigma2)))
        # print(torch.mean((batch_node_value-pre_node_mu)
        #                  ** 2), torch.mean((batch_edge_value-pre_edge_mu)**2))
        if self.use_weight:
            node_loss = torch.mean(1.0/2*torch.log(self.eps+pre_node_sigma2) +
                                   (batch_node_value[:, 0]-pre_node_mu)**2/(self.eps+pre_node_sigma2))
            edge_loss = torch.mean(1.0/2*torch.log(self.eps+pre_edge_sigma2) +
                                   (batch_edge_value[:, 0]-pre_edge_mu)**2/(self.eps+pre_edge_sigma2))
        else:
            node_loss = torch.mean((1.0/2*torch.log(self.eps+pre_node_sigma2) +
                                    (batch_node_value[:, 0]-pre_node_mu)**2/(self.eps+pre_node_sigma2))*batch_node_value[:, 1])
            edge_loss = torch.mean((1.0/2*torch.log(self.eps+pre_edge_sigma2) +
                                    (batch_edge_value[:, 0]-pre_edge_mu)**2/(self.eps+pre_edge_sigma2))*batch_edge_value[:, 1])

        return node_loss, edge_loss, node_loss*self.lamb+edge_loss*(1-self.lamb)


class TestNLLLoss(nn.Module):
    def __init__(self, eps=1, lamb=0.5, kind=0, use_weight=False):
        super(TestNLLLoss, self).__init__()
        self.eps = eps
        self.lamb = lamb
        self.kind = kind
        self.use_weight = use_weight

    def forward(self, ins, target):
        n_mu, n_sigma2, e_mu, e_sigma2 = ins
        batch_node_key, batch_node_value, batch_edge_key, batch_edge_value = target
        pre_node_mu = n_mu.take(batch_node_key)
        pre_node_sigma2 = n_sigma2.take(batch_node_key)
        pre_edge_mu = e_mu.take(batch_edge_key)
        pre_edge_sigma2 = e_sigma2.take(batch_edge_key)
        node_loss = torch.mean(1.0/2*torch.log(self.eps+pre_node_sigma2) +
                               (batch_node_value[:, 0]-pre_node_mu)**2/(self.eps+pre_node_sigma2)) if self.use_weight is False else torch.mean((1.0/2*torch.log(self.eps+pre_node_sigma2) +
                                                                                                                                                (batch_node_value[:, 0]-pre_node_mu)**2/(self.eps+pre_node_sigma2))*batch_node_value[:, 1])
        edge_loss = torch.mean(1.0/2*torch.log(self.eps+pre_edge_sigma2) +
                               (batch_edge_value[:, 0]-pre_edge_mu)**2/(self.eps+pre_edge_sigma2)) if self.use_weight is False else torch.mean((1.0/2*torch.log(self.eps+pre_edge_sigma2) +
                                                                                                                                                (batch_edge_value[:, 0]-pre_edge_mu)**2/(self.eps+pre_edge_sigma2))*batch_edge_value[:, 1])
        if self.kind == 1:
            return node_loss
        elif self.kind == 2:
            return edge_loss
        else:
            return node_loss*self.lamb+edge_loss*(1-self.lamb)


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
        if type(in_dim) is tuple:
            n_in_dim, e_in_dim = in_dim
        else:
            n_in_dim, e_in_dim = in_dim, in_dim
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = n_in_dim, n_in_dim
        self._out_feats = out_dim
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc_n = nn.Linear(n_in_dim, out_dim*num_heads, bias=False)
        self.fc_e = nn.Linear(e_in_dim, out_dim*num_heads, bias=False)

        self._cat = cat
        self._edge_type_num = edge_type_num
        self._node_feature_dim = node_feature_dim
        self._edge_feature_dim = edge_feature_dim

        self.use_static_feature = use_static_feature

        if self.use_static_feature:
            self.attn_l = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, out_dim+node_feature_dim, self._edge_type_num)))
            self.attn_r = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, out_dim+node_feature_dim, self._edge_type_num)))
            self.attn_m = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, out_dim+edge_feature_dim, self._edge_type_num)))
        else:
            self.attn_l = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, out_dim, self._edge_type_num)))
            self.attn_r = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, out_dim, self._edge_type_num)))
            self.attn_m = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, out_dim, self._edge_type_num)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_dim:
                self.res_fc_n = nn.Linear(
                    self._in_dst_feats, num_heads * out_dim, bias=False)
                self.res_fc_e = nn.Linear(
                    self._in_dst_feats, num_heads * out_dim, bias=False)
            else:
                self.res_fc_n = Identity()
                self.res_fc_e = Identity()

        else:
            self.register_buffer('res_fc_n', None)
            self.register_buffer('res_fc_e', None)
        self.reset_parameters()
        self.activation = None if activation == 'none' else (
            nn.ReLU() if activation == 'relu' else nn.Sigmoid())

    def edge_attn_udf(self, edges):
        left = torch.bmm(
            edges.src['el'], edges.data['mask'].view(-1, self._edge_type_num, 1))
        right = torch.bmm(
            edges.dst['er'], edges.data['mask'].view(-1, self._edge_type_num, 1))
        return {'e': left+right+edges.data['em']}

    def edge_mean_udf(self, edges):  # 使用三个求平均
        return {'ft': (edges.data['ft']+edges.src['ft']+edges.dst['ft'])/3}

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
                e_feat = nn.Parameter(
                    graph.edata['raw_pro'], requires_grad=False)
            else:
                n_feat, e_feat = feat
            # _, _ = n_feat.shape
            # print(n_feat.shape, e_feat.shape)

            h_src = h_dst = self.feat_drop(n_feat)
            h_edge = self.feat_drop(e_feat)

            feat_src = feat_dst = self.fc_n(h_src).view(
                -1, self._num_heads, self._out_feats)

            feat_edge = self.fc_e(h_edge).view(
                -1, self._num_heads, self._out_feats)

            e_mask = graph.edata['mask'].view(-1, self._edge_type_num, 1)
            if self.use_static_feature:
                n_feat = graph.ndata['pro'].unsqueeze(1).repeat(
                    1, self._num_heads, 1)

                e_feat = graph.edata['pro'].unsqueeze(1).repeat(
                    1, self._num_heads, 1)

                # .unsqueeze(1).repeat(1,batch_size*self._num_heads,1)

                el = (torch.cat([feat_src, n_feat], dim=-1).unsqueeze(3).repeat(1, 1, 1, self._edge_type_num)
                      * self.attn_l).sum(dim=-2).view(-1, self._num_heads, self._edge_type_num)
                er = (torch.cat([feat_dst, n_feat], dim=-1).unsqueeze(3).repeat(
                    1, 1, 1, self._edge_type_num)*self.attn_r).sum(dim=-2).view(-1, self._num_heads, self._edge_type_num)
                em = (torch.cat([feat_edge, e_feat], dim=-1).unsqueeze(3).repeat(1, 1, 1, self._edge_type_num) *
                      self.attn_m).sum(dim=-2).view(-1, self._num_heads, self._edge_type_num)

            else:
                el = (feat_src.unsqueeze(3).repeat(1, 1, 1, self._edge_type_num) *
                      self.attn_l).sum(dim=-2).view(-1, self._num_heads, self._edge_type_num)
                er = (feat_dst.unsqueeze(3).repeat(1, 1, 1, self._edge_type_num) *
                      self.attn_r).sum(dim=-2).view(-1, self._num_heads, self._edge_type_num)
                em = (feat_edge.unsqueeze(3).repeat(1, 1, 1, self._edge_type_num) *
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

            graph.edata.update({'ft': feat_edge})
            graph.apply_edges(self.edge_mean_udf)

            rst_e = graph.edata.pop('ft')

            # residual
            if self.res_fc_n is not None and self.res_fc_e is not None:
                resval_n = self.res_fc_n(h_dst).view(
                    h_dst.shape[0], -1, self._out_feats)
                resval_e = self.res_fc_e(h_edge).view(
                    h_edge.shape[0], -1, self._out_feats)
                rst_n = rst_n + resval_n
                rst_e = rst_e + resval_e

            if self._cat:
                rst_n = rst_n.view(-1, self._num_heads *
                                   self._out_feats)  # 最后用于改变结果
                rst_e = rst_e.view(-1, self._num_heads*self._out_feats)
            else:
                rst_n = rst_n.view(-1, self._num_heads,
                                   self._out_feats).mean(1)  # 不改维度
                rst_e = rst_e.view(-1, self._num_heads,
                                   self._out_feats).mean(1)  # 不改变
            # activation
            if self.activation:
                rst_n = self.activation(rst_n)
                rst_e = self.activation(rst_e)
            return rst_n, rst_e

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_n.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        # print(self.attn_l.shape)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_m, gain=gain)
        if isinstance(self.res_fc_n, nn.Linear):
            nn.init.xavier_normal_(self.res_fc_n.weight, gain=gain)
            nn.init.xavier_normal_(self.res_fc_e.weight, gain=gain)


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
        # self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.i_gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATLayer(
            in_dim, num_hidden, heads[0], edge_type_num,
            feat_drop, attn_drop, negative_slope, False, self.activation, node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))
        self.i_gat_layers.append(GATLayer(
            in_dim, num_hidden, heads[0], edge_type_num,
            feat_drop, attn_drop, negative_slope, False, self.activation, node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            # print(l, num_layers, heads)
            self.gat_layers.append(GATLayer(
                num_hidden * heads[l-1], num_hidden, heads[l], edge_type_num,
                feat_drop, attn_drop, negative_slope, residual, self.activation, node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))
            self.i_gat_layers.append(GATLayer(
                num_hidden * heads[l-1], num_hidden, heads[l], edge_type_num,
                feat_drop, attn_drop, negative_slope, residual, self.activation, node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))
        # output projection
        self.gat_layers.append(GATLayer(
            num_hidden * heads[-2], out_dim, heads[-1], edge_type_num,
            feat_drop, attn_drop, negative_slope, residual, None, cat=False, node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))
        self.i_gat_layers.append(GATLayer(
            num_hidden * heads[-2], out_dim, heads[-1], edge_type_num,
            feat_drop, attn_drop, negative_slope, residual, None, cat=False, node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim, use_static_feature=use_static_feature))

    def forward(self, g, inv_g):
        # h = inputs
        h = None
        # print("ok1")
        for l in range(self.num_layers):
            h_n, h_e = self.gat_layers[l](g, h)
            ih_n, ih_e = self.i_gat_layers[l](inv_g, h)
            h = (h_n+ih_n, h_e+ih_e)
            # print(l, h[0].shape, h[1].shape)
        # output projection
        h_n, h_e = self.gat_layers[-1](g, h)
        ih_n, ih_e = self.i_gat_layers[-1](inv_g, h)
        # print(h_n.shape, h_e.shape)
        # logits = self.gat_layers[-1](g, h) + self.i_gat_layers[-1](inv_g, h)
        return (h_n+ih_n, h_e+ih_e)

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
        # 边的id转到index（数值tensor的第一维度）
        self.edge_id2index_dict = args['edge_id2index_dict']
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
        node_mintime = []
        for i in range(len(self.edge_graph.nodes)):
            v = self.edge_graph.nodes[i]
            nodes_pro.append(v)
            node_mintime.append(v[-2]/v[-1]*60.0)
        g = dgl.graph((torch.tensor(source_list), torch.tensor(
            target_list)), num_nodes=len(self.edge_dict))

        inv_g = dgl.graph((torch.tensor(target_list), torch.tensor(
            source_list)), num_nodes=len(self.edge_dict))
        print(type(g), type(inv_g))

        g.ndata['raw_pro'] = torch.FloatTensor(nodes_pro)  # 赋予节点相关的属性
        inv_g.ndata['raw_pro'] = torch.FloatTensor(nodes_pro)

        g.edata['raw_pro'] = torch.FloatTensor(edge_pros)
        g.edata['mask'] = torch.FloatTensor(edge_mask)

        inv_g.edata['raw_pro'] = torch.FloatTensor(edge_pros)
        inv_g.edata['mask'] = torch.FloatTensor(edge_mask)
        # inv_g.edata['mintime'] = torch.FloatTensor(edge_mintime)

        g.ndata['mintime'] = torch.FloatTensor(node_mintime)

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
            nodes[self.edge_id2index_dict[k]] = v
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
            edges[v] = (type_turning, (self.edge_id2index_dict[e1],
                                       self.edge_id2index_dict[e2]), e1_pro+e2_pro+n_pro)
            # 节点出度统计
            out_dict[self.edge_id2index_dict[e1]] = out_dict.get(
                self.edge_id2index_dict[e1], []) + [self.edge_id2index_dict[e2]]
            # 节点入度统计
            in_dict[self.edge_id2index_dict[e2]] = in_dict.get(
                self.edge_id2index_dict[e2], [])+[self.edge_id2index_dict[e1]]
            # 节点tuple对应的边统计
            node2edge_dict[(self.edge_id2index_dict[e1],
                            self.edge_id2index_dict[e2])] = v
        return EdgeGraph._make([nodes, edges, in_dict, out_dict, node2edge_dict])


class Predictor(nn.Module):
    def __init__(self, in_feat, out_feat, activation):
        self.predictor = nn.Linear(in_feat, out_feat)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.predictor(x))


class ArGenerator(nn.Module):
    def __init__(self, generator, n_grucell, e_grucell, hidden_dim):
        self.generator = generator
        self.n_grucell = n_grucell
        self.e_grucell = e_grucell
        self.n_predictor = Predictor(hidden_dim, 1, nn.Sigmoid)
        self.e_predictor = Predictor(hidden_dim, 1, nn.ReLU)
        self.n_predictor_sigma2 = Predictor(hidden_dim, 1, nn.ReLU)
        self.e_predictor_sigma2 = Predictor(hidden_dim, 1, nn.ReLU)

    def forward(self, g, in_feat, hid_feat):
        x_n, x_e = self.generator(g, in_feat)
        _, batch_size, feat_dim = x_n.shape
        _, _, hid_dim = hid_feat
        h_n = self.n_grucell(
            x_n.view(-1, feat_dim), hid_feat.view(-1, hid_dim)).view(-1, batch_size, hid_dim)
        h_e = self.e_grucell(
            x_e.view(-1, feat_dim), hid_feat.view(-1, hid_dim)).view(-1, batch_size, hid_dim)

        return h_n, self.n_predictor(h_n), h_e, self.e_predictor(h_e)


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.lr = nn.Linear(10, 1)

    def forward(self, x):
        return self.lr(x)


class LeimPatcher(nn.Module):
    def __init__(self, args):
        super(LeimPatcher, self).__init__()
        # out_dim = args['out_dim']

        # 时间信息
        self.slot_size = args['slot_size']
        self.time_embedding_dim = args['time_embedding_size']
        self.day_embedding_dim = args['day_embedding_size']
        self.device = args['device']

        self.day_in_week_embedding = nn.Embedding(7, self.day_embedding_dim)
        self.time_in_day_embedding = nn.Embedding(
            int(24*60/self.slot_size), self.time_embedding_dim)
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
        # patcher_gat_param = args['patcher_gat_param']
        # patcher_gat_param['edge_type_num'] = len(self.graph.turning_types)
        args['patcher_gat_param']['in_dim'] = (
            self.edge_pro_dim, 2*self.edge_pro_dim+self.node_pro_dim)
        self.patcher_gat_param = GATParam(**args['patcher_gat_param'])
        print(self.patcher_gat_param)

        # 道路属性的编码（meta-learner）
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

        # 构建patcher
        self.patcher_gat = GAT(*tuple(self.patcher_gat_param))
        # self.generator_gat_param = args['generator_gat_param']
        # self.generator_gat = GAT(*tuple(self.generater_gat_param))

        # # 构建encoder中的GRU
        # self.grucell_param = GRUCellParam(**args['patcher_grucell_param'])
        # self.grucell = nn.GRUCell(*tuple(self.grucell_param))
        # # 构建meta gru (从node和edge feat修饰 输入和hidden)
        # self.nodefeat2input = nn.Parameter(torch.zeros(
        #     size=(self.patcher_gat_param.node_feature_dim, self.grucell_param.input_size * self.patcher_gat_param.out_dim)))
        # nn.init.xavier_uniform_(self.nodefeat2input.data, gain=1.414)
        # self.edgefeat2input = nn.Parameter(torch.zeros(
        #     size=(self.patcher_gat_param.edge_feature_dim, self.grucell_param.input_size * self.patcher_gat_param.out_dim)))
        # nn.init.xavier_uniform_(self.edgefeat2input.data, gain=1.414)
        # self.nodefeat2hidden = nn.Parameter(torch.zeros(
        #     size=(self.patcher_gat_param.node_feature_dim, self.grucell_param.hidden_size * self.grucell_param.hidden_size)))
        # nn.init.xavier_uniform_(self.nodefeat2input.data, gain=1.414)
        # self.edgefeat2hidden = nn.Parameter(torch.zeros(
        #     size=(self.patcher_gat_param.edge_feature_dim, self.grucell_param.hidden_size * self.grucell_param.hidden_size)))
        # nn.init.xavier_uniform_(self.edgefeat2hidden.data, gain=1.414)

        # assert self.gru_param.input_size == self.patcher_gat_param.out_dim

        # 周期性质和节点特征做矩阵乘法得到基准预测结果
        self.base_n_mu = nn.Parameter(torch.zeros(
            size=(self.patcher_gat_param.out_dim, self.day_embedding_dim+self.time_embedding_dim)))
        nn.init.xavier_uniform_(self.base_n_mu.data, gain=1.414)
        self.base_n_sigma = nn.Parameter(torch.zeros(
            size=(self.patcher_gat_param.out_dim, self.day_embedding_dim+self.time_embedding_dim)))
        nn.init.xavier_uniform_(self.base_n_sigma.data, gain=1.414)
        # Predictor(self.day_embedding_dim+self.time_embedding_dim,1,nn.Sigmoid)
        self.base_e_mu = nn.Parameter(torch.zeros(
            size=(self.patcher_gat_param.out_dim, self.day_embedding_dim+self.time_embedding_dim)))
        nn.init.xavier_uniform_(self.base_e_mu.data, gain=1.414)
        self.base_e_sigma = nn.Parameter(torch.zeros(
            size=(self.patcher_gat_param.out_dim, self.day_embedding_dim+self.time_embedding_dim)))
        nn.init.xavier_uniform_(self.base_e_sigma.data, gain=1.414)

        # self.g = self.g.to(self.device)
        # self.inv_g = self.inv_g.to(self.device)

    def to(self, device):
        super().to(device)
        self.g, self.inv_g = self.g.to(
            device), self.inv_g.to(device)

    def forward(self, x):
        # is_train: 如果是训练的话，那么y是和x具有一样的表示，都是稀疏的分布表示
        '''input: encoder_x: (day: Tensor([batch_size]), time: Tensor([batch_size)'''
        '''output: node_out: Tensor([batch_size, node_size, out_dim]), edge_out: Tensor([batch_size, edge_size, out_dim])'''

        day, time = x
        # print(type(self.g))

        batch_size = len(day)

        temporal_emb = torch.cat([self.day_in_week_embedding(
            day), self.time_in_day_embedding(time)], dim=-1)  # [batch_size, day_dim+time+dim]

        # 调用patcher获得补全
        # node_h, edge_h = self.patcher_gat(self.g, self.inv_g, (node_feat.view(batch_size*time_steps, node_size, -1).transpose(
        #     1, 0),  edge_feat.view(batch_size*time_steps, edge_size, -1).transpose(1, 0)))

        # node_h = node_h.transponse(1, 0).contiguous().view(
        #     batch_size, node_size, time_steps, -1)
        # edge_h = edge_h.transpose(1, 0).contiguous().view(
        #     batch_size, edge_size, time_steps, -1)
        # 编码得到dgl图点边的属性
        self.g.ndata['pro'] = self.edge_weight_nn(self.g.ndata['raw_pro'])
        self.inv_g.ndata['pro'] = self.edge_weight_nn(
            self.inv_g.ndata['raw_pro'])
        self.g.edata['pro'] = self.turning_weight_nn(self.g.edata['raw_pro'])
        self.inv_g.edata['pro'] = self.turning_weight_nn(
            self.inv_g.edata['raw_pro'])

        node_h, edge_h = self.patcher_gat(self.g, self.inv_g)
        # 过一下GRU
        # node_h, edge_h = self.gruprocess(self.g, (node_h, edge_h), None)

        # 基准
        n_mu = torch.bmm(
            torch.matmul(node_h, self.base_n_mu).unsqueeze(0).repeat(batch_size, 1, 1), temporal_emb.view(batch_size, -1, 1)).view(batch_size, self.node_size)

        # n_mu = self.g.ndata['mintime'].unsqueeze(
        #     0).repeat(batch_size, 1)/n_mu  # 考虑到最大的

        n_sigma2 = torch.bmm(
            torch.matmul(node_h, self.base_n_sigma).unsqueeze(0).repeat(batch_size, 1, 1), temporal_emb.view(batch_size, -1, 1)).view(batch_size, self.node_size)

        e_mu = torch.bmm(
            torch.matmul(edge_h, self.base_e_mu).unsqueeze(0).repeat(batch_size, 1, 1), temporal_emb.view(batch_size, -1, 1)).view(batch_size, self.edge_size)

        e_sigma2 = torch.bmm(
            torch.matmul(edge_h, self.base_e_sigma).unsqueeze(0).repeat(batch_size, 1, 1), temporal_emb.view(batch_size, -1, 1)).view(batch_size, self.edge_size)
        # print(n_mu[:, :100], n_sigma[:, :100], e_mu[:, :100], e_sigma[:, :100])
        # print(n_mu.shape)
        return n_mu, n_sigma2**2, e_mu, e_sigma2**2

# def gruprocess(self, g, x, hid):
    #     # x: ([node_size, batch_size, time_steps, -1], [edge_size, batch_size, time_steps, -1]), hid: None
    #     node_pro = g.ndata['pro']
    #     edge_pro = g.edata['pro']
    #     x_node, x_edge = x
    #     batch_size, node_size, time_steps, _ = x_node
    #     _, edge_size, _, _ = x_edge
    #     if hid is None:
    #         hid_node = torch.zeros(
    #             size=(batch_size*node_size, self.grucell_param.hidden_size))
    #         hid_edge = torch.zeros(
    #             size=(batch_size*edge_size, self.grucell_param.hidden_size))
    #     node_output = []
    #     edge_output = []
    #     for i in range(time_steps):
    #         hid_node = self.grucell(torch.bmm(torch.matmul(node_pro, self.nodefeat2input).view(node_size, self.grucell_param.input_size, -1).repeat(batch_size, 1, 1), x_node[:, :, i, :].view(-1, self.patcher_gat_param.out_dim, 1)).view(-1, self.grucell_param.input_size), torch.bmm(
    #             torch.matmul(node_pro, self.nodefeat2hidden).view(node_size, self.grucell_param.hidden_size, -1).repeat(batch_size, 1, 1), hid_node.view(-1, self.grucell_param.hidden_size, 1)).view(-1, self.grucell_param.hidden_size))

    #         hid_edge = self.grucell(torch.bmm(torch.matmul(edge_pro, self.edgefeat2input).view(edge_size, self.grucell_param.input_size, -1).repeat(batch_size, 1, 1), x_edge[:, :, i, :].view(-1, self.patcher_gat_param.out_dim, 1)).view(-1, self.grucell_param.input_size), torch.bmm(
    #             torch.matmul(edge_pro, self.edgefeat2hidden).view(edge_size, self.grucell_param.hidden_size, -1).repeat(batch_size, 1, 1), hid_edge.view(-1, self.grucell_param.hidden_size, 1)).view(-1, self.grucell_param.hidden_size))

    #         node_output.append(hid_node)
    #         edge_output.append(hid_edge)

    #     return torch.stack(node_output, dim=1).view(batch_size, node_size, time_steps, -1), torch.stack(edge_output, dim=1).view(batch_size, edge_size, time_steps, -1)


'''实现一个时空融合的transformer编码和解码器得到每条路段以及每个交叉口的通勤时间'''

STParam = namedtuple('STParam', ['embedding_size_1', 'lens', 'embedding_size_2', 'hidden_size', 'num_layers', 'num_heads', 'total_key_depth', 'total_value_depth',
                                 'filter_size', 'max_length', 'input_dropout', 'layer_dropout',
                                 'attention_dropout', 'relu_dropout', 'use_mask', 'act', 'kernel_size', 'wide', 'height'])

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
                 attention_dropout, relu_dropout, use_mask, act, kernel_size, wide, height):
        super(STTransformer, self).__init__()
        # self.encoder_input_size = embedding_size #
        self.transformer_enc = Encoder(embedding_size_1, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                                       filter_size,  lens, input_dropout, layer_dropout,
                                       attention_dropout, relu_dropout, act, kernel_size, wide, height)

        self.transformer_dec = Decoder(embedding_size_2, hidden_size, num_layers, num_heads, total_key_depth, total_key_depth, filter_size,
                                       max_length, input_dropout, layer_dropout, attention_dropout, relu_dropout, act, kernel_size, wide, height, use_mask)

        self.generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Rel(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x_enc, x_dec, src_mask, n_mu, n_sigma2, e_mu, e_sigma2, n_min_time):
        # x_enc: [batch_size, enc_seq_length, w, h, 2]
        # x_dec: [batch_size, dec_seq_length, node_edge_embedding]
        # src_mask: [batch_size, dec_seq_length]
        # n_sigma2, n_mu, n_min_time: [batch_size, dec_node_length]
        # e_mu, e_sigma2: [batch_size, dec_edge_length]

        encoder_output = self.transformer_enc(x_enc)
        decoder_output = self.transformer_dec(x_dec, encoder_output, src_mask)
        src_node_mask = src_mask[:, ::2]
        src_edge_mask = src_mask[:, 1::2]
        decoder_node = (n_min_time /
                        torch.sigmoid(decoder_output[:, :, 0][:, ::2]*n_sigma2+n_mu)).masked_fill(src_node_mask == 0, 0)

        decoder_edge = torch.relu(
            decoder_output[:, :, 1][:, 1::2]*e_sigma2+e_mu).masked_fill(src_edge_mask == 0, 0)

        return decoder_node, src_node_mask, decoder_edge, src_edge_mask


class Leim(nn.Module):
    def __init__(self, args):
        # self._pather = LeimPatcher(args)  # 获得所有的link对应的时间信息，然后需要根据具体的link表示抽取出特定的
        self._pather = LeimPatcher(args)

        stParam = STParam(**args['stparam'])
        self._transformer = STTransformer(*tuple(stParam))

        # 将edge和node映射到同一维度上去
        # 道路属性的编码（meta-learner）
        self.node_emb = nn.Linear(
            args['g'].ndata['raw_pro'].shape[-1], stParam.embedding_size_2),
        # 道路交叉点使用两条边加中间节点的属性信息（meta-learner）
        self.edge_emb = nn.Linear(
            args['g'].edata['raw_pro'].shape[-1], stParam.embedding_size_2)

        self.mintime = nn.Parameter(
            args['g'].edata['mintime'], requires_grad=False)

    def forward(self, d_t, x_idx, x, src_mask):
        # x_enc: [batch_size, enc_seq_length, w, h, 2]
        # node_dec: [batch_size, dec_node_length, raw_node_pro_dim]
        # edge_dec: [batch_size, dec_edge_length, raw_edge_pro_dim]
        # node_mask: [bath_size, dec_node_length]
        # edge_mask: [batch_size, dec_edge_length]
        n_mu, n_sigma2, e_mu, e_sigma2 = self._pather(d_t)

        x_enc, node_dec, edge_dec = x
        bs, node_len, _ = node_dec.shape
        _, edge_len, _ = edge_dec.shape
        node_dec = self.node_emb(node_dec)
        edge_dec = self.edge_emb(edge_dec)
        x_dec = torch.cat([node_dec, edge_dec], dim=1).permute(
            1, 0, 2).contiguous().view(node_len+edge_len, -1)
        idx = torch.LongTensor(
            [i//2 if i % 2 == 0 else i//2+node_len for i in range(node_len+edge_len)])
        x_dec = x_dec.index_select(index=idx, dim=0).view(
            node_len+edge_len, bs, -1).permute(1, 0, 2).contiguous()

        n_idx, e_idx = x_idx
        n_mu = torch.stack([n_mu[i].take(n_idx[i]) for i in range(bs)], dim=0)
        n_sigma2 = torch.stack([n_sigma2[i].take(n_idx[i])
                                for i in range(bs)], dim=0)
        mintime = torch.stack([self.mintime.take(n_idx[i])
                               for i in range(bs)], dim=0)
        e_mu = torch.stack([e_mu[i].take(e_idx[i]) for i in range(bs)], dim=0)
        e_sigma2 = torch.stack([e_sigma2[i].take(e_idx[i])
                                for i in range(bs)], dim=0)

        decoder_node, src_node_mask, decoder_edge, src_edge_mask = self._transformer(
            x_enc, x_dec, src_mask, n_mu, n_sigma2, e_mu, e_sigma2, mintime)
        return decoder_node, src_node_mask, decoder_edge, src_edge_mask
