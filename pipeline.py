# -*- encoding:utf-8 -*-
# 通过python
import torch
import torch.nn as nn
import argparse
import os
import pandas as pd
import imp
import math
from numpy import array, zeros, argmin, inf, ndim
import matplotlib.pyplot as plt
from scipy import interpolate
from shapely.geometry import LineString
import json
import itertools
import re
from datetime import datetime
import time
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, _prepare_batch
from tqdm import tqdm
import torch.nn.functional as F
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
import glob
import yaml
import pickle
import random
import numpy as np
from datetime import timedelta
from ignite.handlers import EarlyStopping, ModelCheckpoint
from torch.optim import SGD, Adam, RMSprop
from shapely.geometry import Polygon, MultiPolygon, MultiPoint, LineString, Point
import functools

import sys
# change to your own directory
proj_dir = "./"
sys.path.append(proj_dir+"code/")
import commonLayers  # NOQA:E402
import Leim2  # NOQA:E402

# from torchvision.transforms import Compose, ToTensor, Normalize
seed = 10
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser(description='Process some integers.')

# static configuration parameters
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--city', type=str, required=False,
                    default='chengdu', help='the kind of dataset')

parser.add_argument('--yaml_file', type=str, required=True,
                    help='the file of configurations')


# dynamic configuration parameters (for robustness)
parser.add_argument('--slot_size', type=int, default=5,
                    help='the actual slot size')
parser.add_argument('--pre_slot_num', type=int, default=12,
                    help='the actual number of past time slots')

# hyper-parameters
parser.add_argument('--split_scale', type=float, default=0.5,
                    help='scale the size of each historgram')
parser.add_argument('--global_num_hidden', type=int, default=16,
                    help='the hidden size of the global static histogram gat, also influencing the out_dim when we set the head number as 3')
parser.add_argument('--node_feature_dim', type=int, default=32,
                    help='the hidden size of the feature embedding')
parser.add_argument('--local_num_hidden', type=int, default=8,
                    help='the hidden size of the local gat encoding')
parser.add_argument('--st_hidden_size', type=int, default=64,
                    help='the hidden size of the output code')
parser.add_argument('--st_total_key_depth', type=int,
                    default=48, help='the hidden size for the key or query')
parser.add_argument('--st_total_value_depth', type=int,
                    default=48, help='the hidden size for the value')
parser.add_argument('--day_embedding_dim', type=int,
                    default=3, help='the embedding size of each day')
parser.add_argument('--time_embedding_dim', type=int,
                    default=5, help='the embedding size of each time slot')

# training parameters
parser.add_argument('--train_batch_size', type=int, default=500)
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--early_stop_epoch', type=int, default=20)
parser.add_argument('--lamb', type=float, default=0.5)
parser.add_argument('--alpha_1', type=float, default=0.1)
parser.add_argument('--alpha_2', type=float, default=0.1)

parser.add_argument('--scalability', type=float, default=1.0)

parser.add_argument('--evaluate_model', type=bool, default=False)

parser.add_argument('--use_global_gat', type=bool, default=True)
parser.add_argument('--use_local_gat', type=bool, default=True)
parser.add_argument('--use_encoder_attention', type=bool, default=True)
parser.add_argument('--use_decoder_attention', type=bool, default=True)


parser = parser.parse_args()
print(parser)

scalability = parser.scalability
# conf
city = parser.city
start_day = datetime(2016, 10, 1, 0, 0, 0)
end_day = datetime(2016, 11, 30, 23, 59, 59)
raw_edge_split = [6.6, 9.8, 12.9, 15.8, 18.6, 21.2, 23.7, 26,
                  28.2, 30.4, 32.5, 34.7, 37, 39.4, 42, 45, 48.9, 53.9, 63.1, 186]
data_dir = proj_dir+"data/{}/".format(city)
split_scale = parser.split_scale
edge_split = []
if split_scale > 1:
    pre = 0
    for p in raw_edge_split:
        delta = (p-pre)/split_scale
        for j in range(split_scale-1):
            c_ = round(pre+delta*(j+1), 2)
            edge_split.append(c_)
        edge_split.append(p)
        pre = p
else:
    delta_step = int(1/split_scale)
    for i in range(0, len(raw_edge_split), delta_step):
        edge_split.append(raw_edge_split[i])
    if edge_split[-1] != raw_edge_split[-1]:
        edge_split.append(raw_edge_split[-1])

# load configuration
node2edge_dict, edge2node_dict, turning_dict, node_res, edge_res, input_edge_dict, output_edge_dict = pickle.load(
    open(data_dir+"new_roadnetwork_with_properties.pk", "rb"))
graph_args = {
    "node2edge_dict": node2edge_dict,
    "edge2node_dict": edge2node_dict,
    "input_dict": input_edge_dict,
    "output_dict": output_edge_dict,
    "node_dict": node_res,
    "edge_dict": edge_res,
    "turning_dict": turning_dict
}

# define custom tag handler


def join(loader, node):
    #     print(node)
    #     seq = loader.construct_sequence(node)
    res = []
    for i in node.value:
        #         print(i)
        if type(i) == yaml.SequenceNode:
            res.append("-".join([str(ii.value) for ii in i.value]))
        else:
            res.append(str(i.value))
    return '_'.join(res)


# register the tag handler
yaml.add_constructor('!join', join)
conf = yaml.load(open(proj_dir+"code/{}".format(parser.yaml_file), "r"))
# 限制一下route的长度，不然很难
max_link_lens = conf['model_para']['stparam']['max_length']
model_name = f'{parser.slot_size}-{parser.pre_slot_num}--{parser.split_scale}-{parser.global_num_hidden}-{parser.node_feature_dim}-{parser.local_num_hidden}-{parser.st_hidden_size}-{parser.st_total_key_depth}-{parser.st_total_value_depth}-{parser.day_embedding_dim}-{parser.time_embedding_dim}-{parser.lr}-{parser.lamb}-{parser.alpha_1}-{parser.alpha_2}'
# setting values for the configuration


# setting model_para
conf['model_para']['device'] = parser.device
conf['model_para']['graph'] = graph_args
conf['model_para']['patcher_gat_param']['num_hidden'] = parser.global_num_hidden
conf['model_para']['patcher_gat_param']['out_dim'] = parser.global_num_hidden*3
conf['model_para']['patcher_gat_param']['node_feature_dim'] = parser.node_feature_dim
conf['model_para']['localgraph_gat_param']['num_hidden'] = parser.local_num_hidden
conf['model_para']['localgraph_gat_param']['out_dim'] = parser.local_num_hidden*3
conf['model_para']['localgraph_gat_param']['node_feature_dim'] = parser.node_feature_dim
conf['model_para']['stparam']['lens'] = parser.pre_slot_num
conf['model_para']['stparam']['slot_size'] = parser.slot_size
conf['model_para']['stparam']['hidden_size'] = parser.st_hidden_size
conf['model_para']['stparam']['total_key_depth'] = parser.st_total_key_depth
conf['model_para']['stparam']['total_value_depth'] = parser.st_total_value_depth
conf['model_para']['stparam']['time_embedding_dim'] = parser.time_embedding_dim
conf['model_para']['stparam']['day_embedding_dim'] = parser.day_embedding_dim

# setting training_para
conf['training_para']['train_batch_size'] = parser.train_batch_size
conf['training_para']['test_batch_size'] = parser.test_batch_size
conf['training_para']['epochs'] = parser.epochs
conf['training_para']['lr'] = parser.lr
conf['training_para']['early_stop_epoch'] = parser.early_stop_epoch
conf['training_para']['lamb'] = parser.lamb
conf['training_para']['alpha_1'] = parser.alpha_1
conf['training_para']['alpha_2'] = parser.alpha_2
conf['training_para']['model_name'] = model_name
conf['training_para']['slot_size'] = parser.slot_size
conf['training_para']['pre_slot_num'] = parser.pre_slot_num
conf['training_para']['evaluate_model'] = parser.evaluate_model


# setting ablation experiments
conf['model_para']['use_global_gat'] = parser.use_global_gat
conf['model_para']['use_local_gat'] = parser.use_local_gat
conf['model_para']['use_encoder_attention'] = parser.use_encoder_attention
conf['model_para']['use_decoder_attention'] = parser.use_decoder_attention

node_size, edge_size = len(edge_res), len(turning_dict)
# load trajectories
slot_size = int(parser.slot_size)
pre_slot_num = int(parser.pre_slot_num)

# 数据读取
train_data_file = os.path.join(
    data_dir, "train", "train-{}-{}.pk".format(slot_size, pre_slot_num))
test_data_file = os.path.join(
    data_dir, "test", "test-{}-{}.pk".format(slot_size, pre_slot_num))

random.seed(slot_size+pre_slot_num)

print(train_data_file, test_data_file)
print(f"{city}-############{model_name}############")

train_x, train_y = pickle.load(open(train_data_file, "rb"))


test_x, test_y = pickle.load(open(test_data_file, "rb"))

# 读取不同区域的流量数据


class MyDataset(Dataset):  # 构建一周的数据
    def __init__(self, x, y, scalability=1):  # valid_slot之前的都不看
        '''trajs: list(list(flag[0 for edge and 1 for turning, raw_id, time_slot_index_in_default_slot_size, time])),
           traffic_state: np.array([time_slot_index_num_in_default_slot_size, width, height, feature_dim]) '''
        actual_lens = int(len(x)*scalability)
        if actual_lens == 0:
            actual_lens = len(x)
        self.x = x[:actual_lens]
        self.y = y[:actual_lens]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        return x, y

    def __len__(self):
        return len(self.x)


def collate_fn(batch):
    #         begin_time = time.time()
    dec_seq_lens = [len(x[1][1]) for x, _ in batch]
    max_seq = max(dec_seq_lens)
    batch_y = []
    batch_day = []
    batch_time = []
#         x_enc = []
    src_mask = []
    node_dec = []
    edge_dec = []
    batch_edge_value = []
    batch_turning_value = []

    batch_enc_src, batch_enc_dst, batch_enc_turning, node_idxs, node_lens = [], [], [], [], []
    batch_enc_edges = []
    node_cum = 0
    edge_cum = 0
#         print("start batch....")
    x_enc_node = []
    for x, y in batch:
        batch_day.append(x[0][0])
        batch_time.append(x[0][1])
#             x_enc.append(x[1][0])
        extra_node = []
        extra_edge_value = []
        for i in range(max_seq-len(x[1][1])):
            extra_node.append(node_size)
            extra_edge_value.append(-1.0)
        node_dec.append(x[1][1]+extra_node)
        batch_edge_value.append(y+extra_edge_value)

        src_mask.append([1 for i in range(len(x[1][1]))] +
                        [0 for i in range(max_seq-len(x[1][1]))])

        src_, dst_, turning_, node_data, local_edges = x[1][0]

        batch_enc_src.append(torch.LongTensor(src_)+node_cum)
        batch_enc_dst.append(torch.LongTensor(dst_)+node_cum)
        batch_enc_turning.append(torch.LongTensor(turning_))
        batch_enc_edges.append(torch.LongTensor(local_edges))
        x_enc_node.append(torch.FloatTensor(node_data))

        node_idxs.append(torch.LongTensor(
            [i for i in range(len(node_data))])+node_cum)
        node_cum += len(node_data)
        node_lens.append(len(node_data))

#         print("end batch....")

    max_node_len = max(node_lens)
    batch_enc_turning = torch.cat(batch_enc_turning, dim=0)
    batch_enc_edges = torch.cat(batch_enc_edges, dim=0)
    node_lens = torch.LongTensor(node_lens)
    x_enc_node = torch.cat(x_enc_node, dim=0)
    batch_enc_src = torch.cat(batch_enc_src, dim=0)
    batch_enc_dst = torch.cat(batch_enc_dst, dim=0)

    node_idxs = torch.stack([torch.cat([t, torch.LongTensor([0 for j in range(
        max_node_len-node_lens[i])])]) for i, t in enumerate(node_idxs)])

    node_masks = torch.stack([torch.LongTensor([1 for j in range(
        node_lens[i])]+[0 for j in range(max_node_len-node_lens[i])]) for i, t in enumerate(node_idxs)])

    assert torch.isnan(batch_enc_turning).any() == False and torch.isinf(
        batch_enc_turning).any() == False
    assert torch.isnan(node_lens).any() == False and torch.isinf(
        node_lens).any() == False
    assert torch.isnan(batch_enc_src).any() == False and torch.isinf(
        batch_enc_src).any() == False
    assert torch.isnan(batch_enc_dst).any() == False and torch.isinf(
        batch_enc_dst).any() == False
    assert torch.isnan(x_enc_node).any() == False and torch.isinf(
        x_enc_node).any() == False
    assert torch.isnan(node_idxs).any() == False and torch.isinf(
        node_idxs).any() == False
    assert torch.isnan(node_masks).any() == False and torch.isinf(
        node_masks).any() == False

    x_enc = (batch_enc_turning, batch_enc_edges, node_lens,
             batch_enc_src, batch_enc_dst, x_enc_node, node_idxs, node_masks)

    batch_day = torch.LongTensor(batch_day)
    batch_time = torch.LongTensor(batch_time)
    d_t = (batch_day, batch_time)
#         x_enc = torch.FloatTensor(x_enc)
    src_mask = torch.ByteTensor(src_mask)

    node_dec = torch.LongTensor(node_dec)
    x = (x_enc, node_dec)

    assert torch.isnan(node_dec).any() == False and torch.isinf(
        node_dec).any() == False

    batch_edge_value = torch.FloatTensor(batch_edge_value)

    assert torch.isnan(batch_edge_value).any() == False and torch.isinf(
        batch_edge_value).any() == False

#         print(batch_edge_value.shape)
#         print("used time: {}".format(time.time()-begin_time))
    return (d_t, x, src_mask), batch_edge_value


def get_data_loaders(mydataset, batch_size):
    loader = DataLoader(mydataset,
                        batch_size=batch_size,
                        collate_fn=collate_fn,
                        num_workers=0,
                        shuffle=False)
    loader._dataset_kind = None
    return loader


try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError(
        "No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def custome_create_supervised_trainer(model, optimizer, grad_clamp, loss_fn,
                                      device=None, non_blocking=False,
                                      prepare_batch=_prepare_batch,
                                      output_transform=lambda x, y, y_pred, losses: [loss.item() for loss in losses]):
    if device:
        print(device)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
#         print("y=",y)
        loss = loss_fn(y_pred, y)
        loss[-1].backward()
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clamp, grad_clamp)
        optimizer.step()
        return output_transform(x, y, y_pred, loss)
    return Engine(_update)


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(logdir=log_dir)
    return writer


def printParams(model):
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("all size of parameters:{}".format(k))
    return k


def run(model, train_dataset, test_dataset, configure,  load_file=True):
    printParams(model)
    training_para = configure['training_para']
    grad_clamp = training_para['grad_clamp']
    early_stop_epoch = training_para['early_stop_epoch']
    log_dir = None if training_para['log_dir'] is None else training_para['log_dir'] + \
        "/"+training_para['pre']+"/"+str(time.time())
    print(log_dir)
    evaluate_model = training_para['evaluate_model']
    log_interval = training_para.get('log_interval', None)
    epochs = training_para['epochs']
    lamb = training_para['lamb']
    alpha_1 = training_para['alpha_1']
    alpha_2 = training_para['alpha_2']

    model_para = configure['model_para']
    device = model_para['device']
    edge_speed_bins = training_para['edge_speed_bins']
    if device != "cpu":
        #         print("device[-1]:", device[-1])
        #         os.environ["CUDA_VISIBLE_DEVICES"] = device[-1]
        os.environ['CUDA_LAUNCH_BLOCKING'] = device[-1]
        torch.cuda.set_device(int(device[-1]))
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    train_loader = get_data_loaders(
        train_dataset, training_para['train_batch_size'])
    test_loader = get_data_loaders(
        test_dataset, training_para['test_batch_size'])
    file_pre = '/'.join([training_para['model_save_dir'], city +
                         "_"+training_para['pre']+"_"+training_para['model_name']+'_'])
    files = glob.glob(file_pre+"*")
#     print(file_pre,files)
    files_suf = [int(f[len(file_pre):f.index(".pt")]) for f in files]

    start_epoch = 0
    if len(files) > 0 and load_file:
        print(files)
        load_f = files[np.argmax(files_suf)]
        model.load_state_dict(torch.load(load_f))

    optimizer = Adam(model.parameters(), lr=training_para['lr'])
#     optimizer = SGD(model.parameters(), lr=training_para['lr'], momentum=training_para['momentum'])
    loss_fn = Leim2.NLLLoss(edge_speed_bins, alpha_1, alpha_2, lamb, device)
    mse_loss_fn = Leim2.TestNLLLoss(edge_speed_bins, 'mse', device)
    mae_loss_fn = Leim2.TestNLLLoss(edge_speed_bins, 'mae', device)
    mape_loss_fn = Leim2.TestNLLLoss(edge_speed_bins, 'mape', device)

#     model.to(device)
    trainer = custome_create_supervised_trainer(
        model, optimizer, grad_clamp, loss_fn, device)
    evaluator = create_supervised_evaluator(model, metrics={"mse_loss": Loss(mse_loss_fn),
                                                            "mae_loss": Loss(mae_loss_fn),
                                                            "mape_loss": Loss(mape_loss_fn)}, device=device)
    RunningAverage(output_transform=lambda x: x[-1]).attach(trainer, 'loss')
    from ignite.contrib.metrics import GpuInfo
    device_name = "cpu" if device == 'cpu' else "gpu"
    GpuInfo().attach(trainer, name=device_name)
    GpuInfo().attach(evaluator, name=device_name)

    # 保存训练阶段的log
    if log_dir is not None and evaluate_model is False:
        writer = create_summary_writer(model, train_loader, log_dir)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(train_loader) + 1
            if iter % log_interval == 0:
                loss = engine.state.output
                writer.add_scalar("training-{}/train_loss".format(
                    training_para['model_name']), loss[-1], engine.state.iteration)
                writer.add_scalar("training-{}/train_link_loss".format(
                    training_para['model_name']), loss[0], engine.state.iteration)
                writer.add_scalar("training-{}/train_path_mse_loss".format(
                    training_para['model_name']), loss[1], engine.state.iteration)
                writer.add_scalar("training-{}/train_path_mape_loss".format(
                    training_para['model_name']), loss[2], engine.state.iteration)

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=['loss'])

    def score_function(engine):
        val_loss = engine.state.metrics['mape_loss']
        return -val_loss

    # 设定early stop
    handler = EarlyStopping(patience=early_stop_epoch,
                            score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    if load_file:
        # 设定每隔两个epoch保存一下模型
        handler2 = ModelCheckpoint(training_para['model_save_dir'], city+"_"+training_para['pre'],
                                   save_interval=None, n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler2, {
            training_para['model_name']: model})

    # 绑定evaluator

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_results(engine):
        #         model.to("cpu")
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        mse_loss = metrics['mse_loss']
        mae_loss = metrics['mae_loss']
        mape_loss = metrics['mape_loss']
        pbar.log_message("Testing Results - Avg mse loss: {:.4f}, Avg mae loss: {:.4f}, Avg mape loss: {:.4f}"
                         .format(mse_loss, mae_loss, mape_loss))
        if log_dir is not None and evaluate_model is False:
            writer.add_scalar("valdation-{}/mse_loss".format(
                training_para['model_name']), mse_loss, engine.state.epoch+start_epoch)
            writer.add_scalar("valdation-{}/mae_loss".format(
                training_para['model_name']), mae_loss, engine.state.epoch+start_epoch)
            writer.add_scalar("valdation-{}/mape_loss".format(
                training_para['model_name']), mape_loss, engine.state.epoch+start_epoch)
        pbar.n = pbar.last_print_n = 0
#         model.to(device)

    if evaluate_model:
        #         model.to("cpu")
        time_1 = time.time()
        trainer.run(train_loader, max_epochs=1)
        time_2 = time.time()
        print("testing:{} seconds".format(time_2-time_1))
        evaluator.run(test_loader)
        time_3 = time.time()
        print("runing evaluater:{} seconds".format(time_3-time_2))
        metrics = evaluator.state.metrics
        mse_loss = metrics['mse_loss']
        mae_loss = metrics['mae_loss']
        mape_loss = metrics['mape_loss']
        print("Test Results - Avg mse loss: {:.4f}, Avg mae loss: {:.4f}, Avg mape loss: {:.4f}"
              .format(mse_loss, mae_loss, mape_loss))
    else:
        trainer.run(train_loader, max_epochs=epochs)
        if log_dir is not None:
            writer.close()


train_dataset = MyDataset(train_x, train_y, scalability=scalability)
test_dataset = MyDataset(test_x, test_y)

# 构建连接图
model_para = conf['model_para']
graph = Leim2.RoadNetwork(model_para['graph'])
g, inv_g = graph.buildDglGraph()
model_para['road_network'] = graph
model_para['g'] = g
model_para['inv_g'] = inv_g
model_para['n_split_dim'] = len(edge_split)
model_para['stparam']['embedding_size_1'] = model_para['localgraph_gat_param']['out_dim']
model_para['node_size'] = node_size
model_para['edge_size'] = edge_size
model = Leim2.Leim(model_para)
suffix = "_" + "-".join([str(k) for k in raw_edge_split] +
                        [str(split_scale), str(scalability)])
if conf['training_para']['model_name'].endswith(suffix) is False:
    conf['training_para']['model_name'] = conf['training_para']['model_name'] + suffix
print(conf['training_para']['model_name'])

conf['training_para']['edge_speed_bins'] = edge_split

print("####### staring running #################")
run(model, train_dataset, test_dataset, conf, False)
