import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Subset
from dataset import train_dataset
def noniid_data_distribution(dataset, num_clients):
    """
    非独立同分布的数据分配方法
    :param dataset: 多模态数据集 (MultiModalDataset)
    :param num_clients: 客户端数量
    :return: 每个客户端的数据子集列表
    """
    # 数据总量和标签
    data_size = len(dataset)
    labels = dataset.labels  # 获取数据集中的标签
    unique_labels = np.unique(labels)  # 获取唯一标签

    # 根据类别组织样本索引
    class_indices = {label: [] for label in unique_labels}
    for index in range(data_size):
        label = labels[index]
        class_indices[label].append(index)

    # 分配样本到客户端
    client_datasets = [[] for _ in range(num_clients)]
    for class_idx in unique_labels:
        indices = class_indices[class_idx]
        random.shuffle(indices)  # 打乱当前类别的索引顺序

        # 按客户端分配
        for i, index in enumerate(indices):
            client_id = i % num_clients  # 确保均匀分配
            client_datasets[client_id].append(index)

    # 转换为客户端的子集
    client_subsets = []
    for client_data in client_datasets:
        client_subsets.append(Subset(dataset, client_data))

    return client_subsets
def iid_data_distribution(dataset, num_clients):
    # 计算每个客户端的数据量
    data_size = len(dataset)
    client_data_size = data_size // num_clients

    # 随机打乱索引
    indices = list(range(data_size))
    random.shuffle(indices)

    # 分配数据到每个客户端
    client_datasets = []
    for i in range(num_clients):
        client_indices = indices[i * client_data_size:(i + 1) * client_data_size]
        client_datasets.append(torch.utils.data.Subset(dataset, client_indices))

    return client_datasets

# 数据分配

