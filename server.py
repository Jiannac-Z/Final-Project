import torch

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_models):
        """
        聚合客户端模型
        """
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([model[key] for model in client_models]).mean(dim=0)
        self.global_model.load_state_dict(global_state)
        print("Aggregated global model.")
        return self.global_model.state_dict()

    def distribute_neighbor_gradients(self, client_gradients, adjacency_matrix):
        """
        分发邻居梯度
        """
        neighbor_gradients = []
        for i, client_grad in enumerate(client_gradients):
            neighbors = [j for j, is_neighbor in enumerate(adjacency_matrix[i]) if is_neighbor]
            gradients = [client_gradients[j] for j in neighbors]
            neighbor_gradients.append(gradients)
        return neighbor_gradients
