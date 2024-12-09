# 初始化全局模型
from GNN import MultimodalGNN
from client import MultimodalFederatedClient
from server import FederatedServer
from distribution import noniid_data_distribution,iid_data_distribution
from dataset import train_dataset,test_loader
from test import evaluate_model
from energy import calculate_model_size,calculate_computation_energy,calculate_transmission_energy,estimate_flops
import numpy as np

def assign_information_volume_to_clients(num_clients,adjacency_matrix, clients):
    neighbor_data_size=0

    for i in range(num_clients):
        # 本地数据量
        local_data_size = clients[i].model_size
        for j in range(num_clients):
            if adjacency_matrix[i][j]!=0:
                neighbor_data_size += clients[j].model_size


        # 通信数据量
        #neighbor_data_size = np.sum(adjacency_matrix[i]) * model_size

        # 总信息量
        clients[i].information_volume = local_data_size + neighbor_data_size
        print(f"User {i}, information {clients[i].information_volume}")

global_model = MultimodalGNN(input_dim=768, hidden_dim=64, output_dim=32).to('cpu')
server = FederatedServer(global_model)
trans_metrics = []
comm_metrics = []
trans_energy=0
comm_energy=0
# 分配数据集
num_clients = 20
num_actions = 2
epsilon = 0.3  # 探索概率
num_to_schedule = 15
client_subsets = iid_data_distribution(train_dataset, num_clients)
clients = [MultimodalFederatedClient(client_id=i, dataset=subset, device='cpu',num_actions=num_actions) for i, subset in enumerate(client_subsets)]

performance_metrics = {
    "round": [],
    "loss": [],
    "accuracy": []
}
# 定义邻接矩阵（客户端之间的图结构）
np.random.seed(42)  # 固定随机种子以便结果可复现
connectivity_prob = 0.1  # 每个客户端有10%的概率与其他客户端连接
adjacency_matrix = (np.random.rand(num_clients, num_clients) < connectivity_prob).astype(int)
np.fill_diagonal(adjacency_matrix, 0)  # 保证自己与自己不连接
#adjacency_matrix = np.ones((num_clients, num_clients)) - np.eye(num_clients)  # 对角线为0，其余为1
# 联邦学习训练
rounds = 500

client_models, client_gradients = [], []
for round in range(rounds):
    client_models, client_gradients = [], []
    new_metric=[]
    print(f"Round {round + 1}")
    scheduled_clients=[clients[i] for i in np.random.choice(num_clients- 1, size=int(15), replace=False)]

    for client in scheduled_clients:
        model_state, local_gradients = client.local_train(epochs=5)
        client_models.append(model_state)
        client_gradients.append(local_gradients)
        trans_energy+=client.trans_energy
        comm_energy+=client.com_energy
        #total_energy = calculate_total_energy(client, model_size, flops)
        #print(f"Client {client.client_id} - Total Energy: {total_energy:.4f} J")
        new_metric.append(adjacency_matrix[client.client_id])
    trans_metrics.append(trans_energy)
    comm_metrics.append(comm_energy)

    # 模型聚合
    global_model_state = server.aggregate(client_models)
    # 分发邻居梯度
    neighbor_gradients = server.distribute_neighbor_gradients(client_gradients, new_metric)
    # 更新客户端模型
    for client in scheduled_clients:
        client.model.load_state_dict(global_model_state)
        client.local_train(global_model_state)
    #avg_loss, accuracy = evaluate_model(global_model, test_loader, device='cpu')
    accuracy = evaluate_model(global_model, test_loader, device='cpu')
    print(f"Global Model Test - Round {round + 1}: Accuracy = {accuracy:.4f}")
    performance_metrics["round"].append(round + 1)
    #performance_metrics["loss"].append(avg_loss)
    performance_metrics["accuracy"].append(accuracy)
np.savetxt('acuracy.txt', performance_metrics["accuracy"], delimiter=',')
np.savetxt('trans_energy.txt', trans_metrics, delimiter=',')
np.savetxt('com_energy.txt', comm_metrics, delimiter=',')
import matplotlib.pyplot as plt

    # 绘制损失曲线
'''plt.figure()
plt.plot(performance_metrics["round"], performance_metrics["loss"], marker='o', label='Loss')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.title('Global Model Loss Over Rounds')
plt.legend()
plt.show()'''

    # 绘制准确率曲线
plt.figure()
plt.plot(performance_metrics["round"], performance_metrics["accuracy"], marker='o', label='Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Global Model Accuracy Over Rounds')
plt.legend()
plt.show()
plt.figure()
plt.plot(range(rounds), trans_metrics, marker='o', label='Total Energy')
plt.xlabel('Round')
plt.ylabel('Energy (Joules)')
plt.title('Energy Consumption Over Rounds')
plt.legend()
plt.show()
plt.figure()
plt.plot(range(rounds), comm_metrics, marker='o', label='Total Energy')
plt.xlabel('Round')
plt.ylabel('Energy (Joules)')
plt.title('Energy Consumption Over Rounds')
plt.legend()
plt.show()

