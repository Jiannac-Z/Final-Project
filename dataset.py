import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pickle
import random


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


# 加载数据
with open('aligned_50.pkl', 'rb') as file:
    dataset = pickle.load(file)

train_dataset = MultiModalDataset(data=dataset['train'], transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# 加载测试数据
test_dataset = MultiModalDataset(data=dataset['test'], transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

