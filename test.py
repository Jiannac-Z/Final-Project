import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def evaluate_model(model, data_loader, device):
    """
    评估模型性能
    :param model: 训练好的模型
    :param data_loader: 测试数据加载器
    :param device: 设备（CPU 或 GPU）
    :return: 测试损失和其他指标
    """
    model.eval()
    target_dim = 768  # 目标特征维度
    total_loss = 0
    total_correct = 0
    total_samples = 0
    correct = 0
    total=0

    with torch.no_grad():
        for batch_audio, batch_vision, batch_text, batch_labels in data_loader:
            if batch_audio.size(0) < 32:
                continue  # 跳过小批次，确保每次处理完整数据

            # 将每个模态特征填充到相同维度
            text_tensor = F.pad(batch_text, (0, target_dim - batch_text.size(-1))) if batch_text.size(
                -1) < target_dim else batch_text
            audio_tensor = F.pad(batch_audio, (0, target_dim - batch_audio.size(-1))) if batch_audio.size(
                -1) < target_dim else batch_audio
            vision_tensor = F.pad(batch_vision, (0, target_dim - batch_vision.size(-1))) if batch_vision.size(
                -1) < target_dim else batch_vision

            # 拼接多模态特征
            combined_features = torch.cat([text_tensor, audio_tensor, vision_tensor], dim=1).to(device)
            labels = batch_labels.to(device)

            # 动态构建图的边
            num_nodes = combined_features.size(0)
            edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)], dtype=torch.long).t().contiguous().to(
                device)

            # 构建图数据
            data = Data(x=combined_features, edge_index=edge_index)
            '''out = model(data)  # 原始输出
                        #out = out.mean(dim=1)  # 平均池化，调整维度
                        out = torch.softmax(out, dim=1)  # 多分类归一化

                        # 预测与正确率计算
                        predicted = torch.argmax(out, dim=1)  # 获取预测类别
                        correct += (predicted == labels).sum().item()  # 统计正确预测数
                        total_samples += labels.size(0)

                        # 计算损失
                        loss = F.mse_loss(out, labels.float())  # 假设目标是浮点值
                        total_loss += loss.item()

                    avg_loss = total_loss / total_samples
                    accuracy = correct / total_samples # 百分比准确率

                    #print(f"Global Model Test: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
                    #return avg_loss, accuracy'''

            # 模型前向传播
            outputs = model(data)  # 原始输出
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        accuracy_results = correct / (total*3)  # 计算准确率
        return accuracy_results



