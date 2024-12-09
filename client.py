import torch
from torch.utils.data import Dataset, DataLoader, Subset
from GNN import MultimodalGNN
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
import numpy as np
from energy import calculate_computation_energy,calculate_transmission_energy,calculate_model_size,estimate_flops
class MultimodalFederatedClient:
    def __init__(self, client_id, dataset, device,num_actions, neighbors=None, lambda_=0.1):
        self.client_id = client_id
        self.dataset = dataset  # 客户端数据集
        self.device = device  # 设备（CPU/GPU）
        self.f = random.randrange(1, 100)
        self.neighbors = neighbors if neighbors else []  # 邻居客户端
        self.lambda_ = lambda_  # 邻居信息权重
        self.b = random.randrange(1, 100) * 1e6  # 带宽: 1 MHz
        self.P_i = random.randrange(1, 10)  # 发射功率: 1 W
        self.g_i = random.randrange(10, 50)  # 信道增益: 30 dB
        self.BN0 = random.randrange(1, 100) * 1e-9  # 噪声功率谱密度: 1 nW/Hz
        self.c = 0.5
        # 初始化图数据和模型
        self.data = self.encode_and_construct_graph()
        self.q_values = np.zeros(num_actions)

        #self.model = MultimodalGNN(self.data.x.shape[1], 64, 32).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.num_actions = num_actions
        self.participation_history = 0  # 记录参与次数
        self.efficiency_history = []  # 记录每次参与后的效率
        self.is_scheduled = False  # 当前轮是否被调度

    def calculate_energy_and_time(self):
        T_i_tran = 0
        E_i_tran = 0

        # 转换信道增益从dB到线性
        g_linear = 10 ** (self.g_i / 10)

        # 计算传输速率 R_i(t)
        R_i = self.b * np.log2(1 + (self.P_i * g_linear) / (self.b * self.BN0))

        T_i_tran= self.model_size / R_i  # 传输时间
        E_i_tran = self.P_i * T_i_tran  # 能量消耗
        return T_i_tran, E_i_tran

    def energy_cost(self):
        """Calculate the computation energy cost."""
        return self.lambda_ * self.f ** 2 * len(self.dataset)

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
        self.model_size = calculate_model_size(self.model)
        self.trans_time, self.trans_energy = self.calculate_energy_and_time()
        self.com_energy = self.energy_cost()
        return data

    def select_action(self, epsilon):
        """
        根据 ε-贪婪策略选择动作
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)  # 随机探索
        else:
            return np.argmax(self.q_values)  # 利用当前最优动作

    def select_action_softmax(self, tau):
        """
        根据Softmax分布选择动作
        :param tau: 温度参数
        :return: 动作编号
        """
        tau = max(tau, 0.01)  # 确保温度参数不小于0.01
        q_values_scaled = self.q_values / tau
        max_q = np.max(q_values_scaled)
        exp_q_values = np.exp(q_values_scaled - max_q)
        probabilities = exp_q_values / np.sum(exp_q_values)

        if not np.isclose(np.sum(probabilities), 1.0) or np.any(probabilities < 0):
            raise ValueError(f"Invalid probability distribution: {probabilities}")

        action = np.random.choice(range(self.num_actions), p=probabilities)
        return action

    def update_q_value(self, action, reward, alpha=0.1, gamma=0.9):
        """
        更新 Q 值，加入折扣因子 gamma
        """
        self.q_values[action] = (1 - alpha) * self.q_values[action] + alpha * (reward + gamma * max(self.q_values))

    def __repr__(self):
        return f"Client {self.client_id}: Q-Values = {self.q_values}, Participation = {self.participation_history}"


    def local_train(self, epochs=5, neighbor_gradients=None):
        """
        本地训练：使用邻居梯度信息更新
        """
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # 本地前向传播和损失计算
            out = self.model(self.data.to(self.device)).squeeze()
            local_loss = F.mse_loss(out, self.y)
            local_loss.backward()

            # 获取本地梯度
            local_gradients = {name: param.grad.clone() for name, param in self.model.named_parameters()}

            # 聚合邻居梯度
            if neighbor_gradients:
                for name, param in self.model.named_parameters():
                    neighbor_contrib = torch.stack([grad[name] for grad in neighbor_gradients]).mean(dim=0)
                    param.grad += self.lambda_ * neighbor_contrib

            self.optimizer.step()
            print(f"Client {self.client_id} - Epoch {epoch}: Loss = {local_loss.item()}")
        return self.model.state_dict(), local_gradients


