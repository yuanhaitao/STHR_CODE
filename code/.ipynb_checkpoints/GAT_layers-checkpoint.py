# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.sparse as sparse
'''support batch learning'''



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, region_size, concat=True, n_head=1):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.n_head = n_head

        self.W = nn.Parameter(torch.zeros(
            size=(in_features, out_features*n_head)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.batch_norm = nn.BatchNorm1d(out_features*n_head)
        self.a = nn.Parameter(torch.zeros(size=(n_head, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # .cuda() if torch.cuda.is_available() else -9e15 * torch.ones((region_size,region_size))
        self.zero_vec = nn.Parameter(-9e15 * torch.ones(
            (region_size, region_size)), requires_grad=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        bs, N, _ = input.size()
        h = self.batch_norm(torch.mm(input.view(-1, self.in_features), self.W)).view(bs, N, self.n_head,
                                                                                     self.out_features).permute(0, 2, 1, 3).contiguous().view(-1, N, self.out_features)
        # N = h.size()[1]

        # a_input = torch.cat([h.repeat(1, 1, N).view(bs*self.n_head, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(-1, self.n_head, N*N, 2 * self.out_features)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1)).view(bs,self.n_head,N,N)

        # zero_vec = self.zero_vec.repeat(bs,self.n_head,1,1)
        adj = adj.repeat(self.n_head, 1, 1, 1).permute(1, 0, 2, 3).contiguous()
        # print(adj.shape,zero_vec.shape,e.shape)
        # print(e.device,adj.device,self.zero_vec.device,zero_vec.device)
        attention = torch.where(adj > 0, self.leakyrelu(torch.matmul(torch.cat([h.repeat(1, 1, N).view(bs*self.n_head, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(
            -1, self.n_head, N*N, 2 * self.out_features), self.a).squeeze(-1)).view(bs, self.n_head, N, N)*adj, self.zero_vec.repeat(bs, self.n_head, 1, 1))
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h.view(bs, self.n_head, N, self.out_features)).permute(
            0, 2, 1, 3).contiguous().view(bs, N, -1)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class HierarchyAttentionLayer(nn.Module):
    '''hetergeneous  fusion for edges and nodes (indgree and outdegree)'''

    def __init__(self, node_in_dim, node_out_dim, edge_in_dim, edge_out_dim,
                 dropout, alpha, )

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.batch_norm = nn.BatchNorm1d(out_features)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()
        edge_weight = adj[adj > 0].to(dv)

        h = self.batch_norm(torch.mm(input, self.W))
        # h: N x out
        # assert not torch.isnan(h).any()
        try:
            assert not torch.isnan(h).any()
        except Exception as err:
            print(self.W.data, self.a.data)
            print("input:{}, adj: {}, edge:{}, h:{}".format(input, adj, edge, h))
            assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()*edge_weight))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size(
            [N, N]), torch.ones(size=(N, 1), device=dv))
        zero_vec = 9e-15 * torch.ones_like(e_rowsum)
        e_rowsum = torch.where(e_rowsum > 0, e_rowsum, zero_vec)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)

        try:
            assert not torch.isnan(h_prime).any()
        except Exception as err:
            print(self.W.data, self.a.data)
            print("input:{}, adj: {}, edge:{}, edge_h:{}, edge_e:{},e_rowsum:{}, h_prime:{}".format(
                input, adj, edge, edge_h, edge_e, e_rowsum, h_prime))
            assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        try:
            assert not torch.isnan(h_prime).any()
        except Exception as err:
            print(self.W.data, self.a.data)
            print("input:{}, adj: {}, edge:{}, edge_h:{}, edge_e:{},e_rowsum:{}, h_prime:{}".format(input, adj, edge,
                                                                                                    edge_h, edge_e,
                                                                                                    e_rowsum, h_prime))
            assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


'''we can batch learn the graph'''


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, alpha, nheads, region_size):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        #self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, region_size=region_size, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        self.attentions = GraphAttentionLayer(
            nfeat, nhid, region_size=region_size, dropout=dropout, alpha=alpha, concat=True, n_head=nheads)
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        self.add_module('attention', self.attentions)
        self.out_att = GraphAttentionLayer(
            nhid * nheads, noutput, region_size=region_size, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = self.attentions(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x  # F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        # print("init SpGAT........")
        self.dropout = dropout

        self.attentions = nn.ModuleList([SpGraphAttentionLayer(nfeat,
                                                               nhid,
                                                               dropout=dropout,
                                                               alpha=alpha,
                                                               concat=True) for _ in range(nheads)])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             noutput,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x  # F.log_softmax(x, dim=1)
