
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from dataclasses import dataclass

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3

EMBEDDING_DIM = 64
NUM_EMBEDDINGS = 512
COMMITMENT_COST = 0.25

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

from vq_vae import VQVAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VQVAE(
    in_channels=3,
    hidden_channels=64,
    embedding_dim=EMBEDDING_DIM,
    num_embeddings=NUM_EMBEDDINGS,
    commitment_cost=COMMITMENT_COST
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
