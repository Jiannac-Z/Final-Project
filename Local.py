import pickle
from transformers import BertTokenizer, BertModel
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# 检查是否有可用的 GPU
device = torch.device('cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# 加载数据
with open('/kaggle/input/aligned-50-pkl/aligned_50.pkl', 'rb') as file:
    dataset = pickle.load(file)
if 'train' in dataset:
    dataset = dataset['train']
text_features, audio_features, visual_features = [], [], []

for text_sample in dataset['text']:
    text_tensor = torch.tensor(text_sample, dtype=torch.float32)
    text_features.append(text_tensor.squeeze())

for audio_sample in dataset['audio']:
    audio_tensor = torch.tensor(audio_sample, dtype=torch.float32)
    audio_features.append(audio_tensor.squeeze())

for vision_sample in dataset['vision']:
    vision_tensor = torch.tensor(vision_sample, dtype=torch.float32)
    visual_features.append(vision_tensor.squeeze())
# 扩展 audio 和 visual 维度
# 对音频和视觉特征进行扩展
pooled_audio_features = [audio.expand(text.size(0), -1) for audio, text in zip(audio_features, text_features)]
pooled_visual_features = [visual.expand(text.size(0), -1) for visual, text in zip(visual_features, text_features)]
node_features = [torch.cat([text, audio, visual], dim=1) for text, audio, visual in zip(text_features, pooled_audio_features, pooled_visual_features)]

# 检查 node_features 是否为空
if node_features:
    x = torch.stack(node_features)
    x = x.view(1284, -1)  # 展平到 [1284, 50 * 793]
else:
    raise ValueError("node_features is empty, check input feature lists for consistency.")
edge_index = []
for i in range(len(node_features) - 1):
    edge_index.append([i, i + 1])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_index = edge_index[:, edge_index.max(dim=0).values < len(node_features)]
unique_edges = edge_index.unique(dim=1)
data = Data(x=x, edge_index=edge_index)
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
input_dim = x.shape[1]
hidden_dim = 64
output_dim = 1
model = MultimodalGNN(input_dim, hidden_dim, output_dim)
# 定义优化器和目标函数
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, capturable=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
y = torch.tensor([label_sample for label_sample in dataset['regression_labels']], dtype=torch.float32)
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)
y = y.to(device)
model = model.to(device)
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data).squeeze()
    out = model(data).squeeze()
    y = y.squeeze()  # 确保 y 的形状与 out 匹配
    loss = F.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}: Loss {loss.item()}')

# 推理
model.eval()
with torch.no_grad():
    predictions = model(data).squeeze()
    print("Predictions:", predictions.cpu().numpy())