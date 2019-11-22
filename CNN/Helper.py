import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

mean = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, mean)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, mean)
])


train_data = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
test_data = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)

# create valid set
valid_size = 0.2
num_train = len(train_data)
indices = list(range(len(train_data)))
np.random.shuffle(indices)
split = int(valid_size)

split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

import torch.nn as nn

nn.CrossEntropyLoss()

import torch.optim as optim

optim.SGD(model.parameters(), lr=0.001)