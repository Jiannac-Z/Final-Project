import pickle
from transformers import BertTokenizer, BertModel
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from torch_geometric.nn import GCNConv
# 检查是否有可用的 GPU
device = torch.device('cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
class MultiModalDataset(Dataset):
    def __init__(self, data, transform=None):
        self.audio_data = data['audio']
        self.vision_data = data['vision']
        self.text_data = data['text']
        self.labels = data['classification_labels']
        self.transform = transform

    def __len__(self):
        return min(len(self.audio_data), len(self.vision_data), len(self.text_data), len(self.labels))

    def __getitem__(self, idx):
        if idx >= len(self.audio_data):
            raise IndexError("Index out of range")
        audio = self.audio_data[idx]
        vision = self.vision_data[idx]
        text = self.text_data[idx]
        label = self.labels[idx]

        if self.transform:
            audio = self.transform(audio)
            vision = self.transform(vision)
            text = self.transform(text)
        label = torch.tensor(label, dtype=torch.long)

        return audio, vision, text, label


class ToTensor:
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)
with open('/kaggle/input/aligned-50-pkl/aligned_50.pkl', 'rb') as file:
    dataset = pickle.load(file)
train_dataset = MultiModalDataset(data=dataset['train'], transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
def noniid_data_distribution(dataset, num_clients):
    # 计算总数据量
    data_size = len(dataset)

    # 动态获取标签
    labels = dataset.labels  # 直接访问 labels 属性
    unique_labels = np.unique(labels)  # 获取唯一标签

    # 统计每个类的数据样本索引
    class_indices = {label: [] for label in unique_labels}
    for index in range(data_size):
        label = labels[index]
        class_indices[label].append(index)

    # 为每个客户端分配样本
    client_datasets = [[] for _ in range(num_clients)]

    # 确保每个客户端都有数据
    for class_idx in unique_labels:
        # 获取当前类别的样本索引
        indices = class_indices[class_idx]
        random.shuffle(indices)  # 打乱索引顺序

        # 将当前类的样本均匀分配给所有客户端
        for i, index in enumerate(indices):
            client_id = i % num_clients  # 确保样本均匀分配到客户端
            client_datasets[client_id].append(index)

    # 将索引转换为Subset对象
    client_subsets = []
    for client_data in client_datasets:
        client_subsets.append(Subset(dataset, client_data))

    return client_subsets


class MultimodalGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultimodalGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class FederatedClient:
    def __init__(self, user_id, data, device):
        self.user_id = user_id
        self.dataset = data
        self.device = device
        self.data = self.encode_and_construct_graph()
        # self.local_weights = self.local_train()

    def encode_and_construct_graph(self):
        text_features, audio_features, vision_features = [], [], []
        train_loader = DataLoader(self.dataset, batch_size=32, shuffle=True)

        # 预定义一个目标维度，以确保所有特征统一
        target_dim = 768

        for batch_audio, batch_vision, batch_text, batch_labels in train_loader:
            if batch_audio.size(0) < 32:
                continue

            # 转换张量并填充到相同大小
            text_tensor = torch.tensor(batch_text, dtype=torch.float32)
            audio_tensor = torch.tensor(batch_audio, dtype=torch.float32)
            vision_tensor = torch.tensor(batch_vision, dtype=torch.float32)
            if text_tensor.size(-1) < target_dim:
                text_tensor = F.pad(text_tensor, (0, target_dim - text_tensor.size(-1)))
            if audio_tensor.size(-1) < target_dim:
                audio_tensor = F.pad(audio_tensor, (0, target_dim - audio_tensor.size(-1)))
            if vision_tensor.size(-1) < target_dim:
                vision_tensor = F.pad(vision_tensor, (0, target_dim - vision_tensor.size(-1)))
            text_features.append(text_tensor)
            audio_features.append(audio_tensor)
            vision_features.append(vision_tensor)
            self.y = torch.tensor(batch_labels, dtype=torch.float32)
        node_features = [torch.cat([text, audio, vision], dim=1) for text, audio, vision in
                         zip(text_features, audio_features, vision_features)]

        # 将特征堆叠成一个张量
        x = torch.stack(node_features)
        self.input_dim = x.shape[-1]  # 使用拼接后的特征维度作为输入维度

        # 构建图的边
        edge_index = torch.tensor([[i, i + 1] for i in range(len(node_features) - 1)],
                                  dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index)
        hidden_dim, output_dim = 64, 32  # 假设的隐藏和输出维度
        self.model = MultimodalGNN(self.input_dim, hidden_dim, output_dim)
        return data

    def local_train(self, epochs=5, lr=0.01):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.data.to(self.device)).squeeze()
            loss = F.mse_loss(out, self.y)  # 确保 y 已在 encode_and_construct_graph 中定义
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}: Loss {loss.item()}', 'User:', self.user_id)
        return self.model.state_dict()
def aggregate_model_weights(client_weights):
    """Aggregate model weights from each client by averaging."""
    averaged_weights = {}
    for key in client_weights[0].keys():
        averaged_weights[key] = sum(client_weight[key] for client_weight in client_weights) / len(client_weights)
    return averaged_weights
input_dim = 768
hidden_dim = 64
output_dim = 32

global_model = MultimodalGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

num_rounds = 10
num_users=10
user_datasets=noniid_data_distribution(train_dataset, num_users)
users = [FederatedClient(user_id=i, data=user_datasets[i],device=device) for i in
             range(num_users - 1)]
for round_num in range(num_rounds):
    client_weights = []
    for user in users:
        client_weights.append(user.local_train())
    # Server aggregates weights
    aggregated_weights = aggregate_model_weights(client_weights)
    # Update global model with aggregated weights
    global_model.load_state_dict(aggregated_weights)
    print(f"Round {round_num + 1} completed.")