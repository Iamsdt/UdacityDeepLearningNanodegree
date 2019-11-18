import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

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
valid_percentage = 0.2

indices = list(range(len(train_data)))