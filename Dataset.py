import math
import random
import torch
import numpy as np

class Dataset:
    def __init__(self, samples, args) -> None:
        self.samples = samples  # 包含所有样本的数据集
        self.batch_size = args.batch_size  # 批次大小
        self.num_batches = math.ceil(len(self.samples) / args.batch_size)  # 批次数量
        self.modalities = args.modalities  # 模态（这里可以是EEG信号通道）
        self.dataset = args.dataset  # 数据集名称

        # 假设每个EEG信号的维度
        self.embedding_dim = args.dataset_embedding_dims[args.dataset][args.modalities]
        self.normalization = args.normalization  # 是否进行标准化

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)  # 获取原始批次
        return self.padding(batch)  # 对批次进行填充

    def raw_batch(self, index):
        # 获取给定索引的原始批次数据
        assert index < self.num_batches, f"batch_idx {index} > {self.num_batches}"
        batch = self.samples[index * self.batch_size : (index + 1) * self.batch_size]
        return batch

    def padding(self, samples):
        # 对样本进行填充操作，确保每个样本的长度一致
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s.eeg_signal) for s in samples]).long()
        mx = torch.max(text_len_tensor).item()

        input_tensor = torch.zeros((batch_size, mx, self.embedding_dim))  # 初始化输入张量
        labels = []
        for i, s in enumerate(samples):
            cur_len = len(s.eeg_signal)  # 获取当前样本的EEG信号长度

            # 处理EEG信号数据
            eeg_data = []
            for ch in s.eeg_signal:  # 假设每个样本包含多个EEG通道数据
                eeg_data.append(torch.tensor(ch))

            # 拼接不同通道的EEG数据
            eeg_data = torch.stack(eeg_data, dim=-1)  # 假设EEG信号是多个通道的时序数据

            # 进行标准化处理
            if self.normalization == 'zscore':
                eeg_data = (eeg_data - eeg_data.mean(dim=0)) / eeg_data.std(dim=0)
            elif self.normalization == 'minmax':
                eeg_data = (eeg_data - eeg_data.min(dim=0)[0]) / (eeg_data.max(dim=0)[0] - eeg_data.min(dim=0)[0])

            input_tensor[i, :cur_len, :] = eeg_data  # 填充到输入张量中

            labels.append(s.label)  # 添加标签

        label_tensor = torch.tensor(labels).long()  # 将标签转为Tensor

        # 返回填充后的数据字典
        data = {
            "text_len_tensor": text_len_tensor,
            "input_tensor": input_tensor,
            "label_tensor": label_tensor,
        }
        return data

    def shuffle(self):
        # 随机打乱数据集
        random.shuffle(self.samples)

    def time_window_slice(self, window_size, overlap=False):
        """
        对EEG信号进行时间窗口切割。
        :param window_size: 每个时间窗口的大小
        :param overlap: 是否允许窗口重叠
        :return: 切割后的窗口数据
        """
        sliced_samples = []
        for sample in self.samples:
            eeg_signal = sample.eeg_signal
            step = window_size if not overlap else window_size // 2
            for start in range(0, len(eeg_signal) - window_size + 1, step):
                end = start + window_size
                windowed_sample = eeg_signal[start:end]
                sliced_samples.append(windowed_sample)
        return sliced_samples
