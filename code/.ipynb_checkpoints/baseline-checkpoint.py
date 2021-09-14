# baseline
import numpy as np
import torch.nn as nn
from Leim import MAPELoss
import torch
# average travel time from similar time bins


class RNN(nn.Module):
    def __init__(self, args):
        self.node_emb = nn.Embedding(
            args['node_size']+1, args['embedding_size'], padding_idx=args['node_size'])
        hidden_size = args['embedding_size']
        relu_dropout = args['dropout']
        day_embedding_dim = args['day_embedding_dim']
        time_embedding_dim = args['time_embedding_dim']
        slot_size = args['slot_size']

        self.day_embedding = nn.Embedding(8, day_embedding_dim)
        self.time_embedding = nn.Embedding(
            int(24*60/slot_size)+1, time_embedding_dim)

        # 道路交叉点使用两条边加中间节点的属性信息（meta-learner）
        self.edge_emb = nn.Embedding(
            args['edge_size']+1, hidden_size, padding_idx=args['edge_size'])

        self.model = nn.GRU(args['embedding_size'],
                            hidden_size, batch_size=True)

        self.edge_generator = nn.Sequential(
            nn.Linear(hidden_size+day_embedding_dim +
                      time_embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(relu_dropout),
            nn.Linear(hidden_size, 1)
        )

        self.turning_generator = nn.Sequential(
            nn.Linear(hidden_size+day_embedding_dim +
                      time_embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(relu_dropout),
            nn.Linear(hidden_size, 1)
        )

    def format(self, x):
        d_t, x, src_mask = x
        # src_mask: [bath_size, dec_node_length+dec_edge_length]
        _, node_dec, edge_dec = x
        d, t = d_t
        d = self.day_embedding_dim(d[:,-1])
        t = self.time_embedding_dim(d[:,-1])



class AverageModel:
    def __init__(self, raw_node_tensors, raw_edge_tensors, default_node_time=None, node_res=None, default_edge_time=None, edge_res=None, time_slot=5):
        self.node_res = {}
        self.edge_res = {}
        day = 7
        self.time_bins = int(24*60/time_slot)
        self.temporal = day*self.time_bins
        if node_res is None:
            self.default_node_time = 0
            for k, v in raw_node_tensors.items():
                temp = self.node_res.get(k, {})
                for k1, v1 in v.items():
                    time_idx = k1 % self.temporal
                    temp_1 = temp.get(time_idx, (0, 0))  # 抽取老的值
                    temp_mean_value = (
                        temp_1[0]*temp_1[1]+sum(v1))/(temp_1[1]+len(v1))
                    self.default_node_time = np.mean(
                        [self.default_node_time]+v1)
                    temp_len = temp_1[1]+len(v1)
                    temp[time_idx] = (temp_mean_value, temp_len)  # update
                self.node_res[k] = temp  # update
        else:
            self.default_node_time = default_node_time
            self.node_res = node_res

        if edge_res is None:
            self.default_edge_time = 0
            for k, v in raw_edge_tensors.items():
                temp = self.edge_res.get(k, {})
                for k1, v1 in v.items():
                    time_idx = k1 % self.temporal
                    temp_1 = temp.get(time_idx, (0, 0))  # 抽取老的值
                    temp_mean_value = (
                        temp_1[0]*temp_1[1]+sum(v1))/(temp_1[1]+len(v1))
                    self.default_edge_time = np.mean(
                        [self.default_edge_time]+v1)
                    temp_len = temp_1[1]+len(v1)
                    temp[time_idx] = (temp_mean_value, temp_len)  # update
                self.edge_res[k] = temp  # update
        else:
            self.default_edge_time = default_edge_time
            self.edge_res = edge_res

        print("the default node time is:{}, the default edge time is:{}".format(
            self.default_node_time, self.default_edge_time))

        self.mseloss = nn.MSELoss()
        self.maeloss = nn.L1Loss()
        self.mapeloss = MAPELoss()

    def generate(self, test_dataset, batch_size):
        pre = []
        target = []
        real_target = []
        mae_loss = []
        mape_loss = []
        mse_loss = []
        for k, (x, y) in enumerate(test_dataset):
            (d_, t_), (_, node_dec, edge_dec) = x
            node_value, edge_value = y
            real_target.append(np.sum(node_value)+np.sum(edge_value))
            time_idx = (d_[-1]*self.time_bins+t_[-1]+1) % self.temporal
            pre_ = 0.0
            for i, node in enumerate(node_dec):
                temp = self.node_res.get(node, None)
                if temp is not None:
                    temp = temp.get(time_idx, None)
                pre_ += (temp[0]
                         if temp is not None else self.default_node_time)
            for i, edge in enumerate(edge_dec):
                temp = self.edge_res.get(edge, None)
                if temp is not None:
                    temp = temp.get(time_idx, None)
                pre_ += (temp[0]
                         if temp is not None else self.default_edge_time)
            target.append(np.sum(node_value)+np.sum(edge_value))
            pre.append(pre_)
            # print(pre, target)
            if len(pre) % batch_size == 0 or k == len(test_dataset)-1:
                # print(np.mean(pre), np.mean(target))
                mae_loss.append(self.maeloss(torch.FloatTensor(
                    pre), torch.FloatTensor(target)).item())
                mape_loss.append(self.mapeloss(torch.FloatTensor(
                    pre), torch.FloatTensor(target)).item())
                mse_loss.append(self.mseloss(torch.FloatTensor(
                    pre), torch.FloatTensor(target)).item())
                pre = []
                target = []
        # print(mse_loss, mape_loss, mse_loss)
        return np.mean(mse_loss), np.mean(mae_loss), np.mean(mape_loss), np.mean(real_target), np.percentile(np.array(mae_loss), [0, 25, 50, 75, 100]), np.percentile(np.array(mape_loss), [0, 25, 50, 75, 100]), np.percentile(real_target, [0, 25, 50, 75, 100])
