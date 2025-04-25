import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.optim import lr_scheduler

import utils

log = utils.get_logger()


class Optim:
    def __init__(self, lr, max_grad_value, weight_decay):
        self.lr = lr
        self.max_grad_value = max_grad_value
        self.weight_decay = weight_decay
        self.params = None
        self.optimizer = None

    def set_parameters(self, params, name):
        self.params = list(params)
        if name == "sgd":
            self.optimizer = optim.SGD(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "adam":
            self.optimizer = optim.Adam(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "adamw":
            self.optimizer = optim.AdamW(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )

    def get_scheduler(self, sch):
        print("Using Scheduler")
        if sch == "reduceLR":
            sched = lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        elif sch == "expLR":
            sched = ExponentialLR(self.optimizer, gamma=0.9)
        return sched

    def step(self):
        if self.max_grad_value != -1:
            clip_grad_value_(self.params, self.max_grad_value)
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def batch_feat(features, lengths, device):
    node_features = []
    batch_size = features.size(0)

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])  # 获取当前样本的特征
    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]

    return node_features


def batch_graphify_label(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, device):
    node_features, edge_index, edge_type = [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_index_lengths = []
    bsz_length_total = sum(lengths)

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])  # 获取当前样本的特征
        # 根据 EEG 的时序特性，构建节点之间的关系边
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        perms_rec_with_label = [(item_rec[0], item_rec[1] + bsz_length_total) for item_rec in perms_rec]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))
        for item, item_rec, item_rec_label in zip(perms, perms_rec, perms_rec_with_label):
            # 在情绪识别中，可能我们不需要发言人信息，而是基于EEG的时间窗构建边
            s = "0"  # 假设所有边是同一类
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))

            # 假设每对节点的边类型都为0（基于EEG的连接性可以通过特征相似度来定义）
            c = "0"
            edge_type.append(edge_type_to_idx[s + c])

    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_type, edge_index_lengths


def edge_perms(length, window_past, window_future):
    """
    Method to construct the edges of a graph (EEG signal time series) considering the past and future window.
    return: list of tuples. tuple -> (vertex(int), neighbor(int))
    """
    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        # 构建窗口范围内的邻接节点（基于时间窗口）
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # 使用全部过去的上下文
            eff_array = array[: min(length, j + window_future + 1)]
        elif window_future == -1:  # 使用全部未来的上下文
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[
                        max(0, j - window_past): min(length, j + window_future + 1)
                        ]

        # 根据窗口范围构建邻接边
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)

