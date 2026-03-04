
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vq_vae import VQVAE

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

if __name__ == '__main__':
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQVAE(
        in_channels=3,
        hidden_channels=64,
        embedding_dim=EMBEDDING_DIM,
        num_embeddings=NUM_EMBEDDINGS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_recon_loss = 0
        total_vq_loss = 0
        total_commitment_loss = 0

        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()

            x_recon, z_e, z_q, indices = model(x)

            recon_loss = F.mse_loss(x_recon, x)
            vq_loss = F.mse_loss(z_q, z_e.detach())
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            loss = recon_loss + vq_loss + COMMITMENT_COST * commitment_loss

            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_commitment_loss += commitment_loss.item()

        n = len(train_loader)
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"recon: {total_recon_loss/n:.4f}  "
            f"vq: {total_vq_loss/n:.4f}  "
            f"commit: {total_commitment_loss/n:.4f}"
        )
