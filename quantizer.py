
import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z_e):
        # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        z = z_e.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)

        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flat, self.codebook.weight.t())
            + torch.sum(self.codebook.weight ** 2, dim=1)
        )

        indices = torch.argmin(distances, dim=1)

        z_q_flat = self.codebook(indices)
        z_q = z_q_flat.view(z.shape)

        # Straight-Through Estimator
        z_q_ste = z + (z_q - z).detach()

        # (B, H, W, C) -> (B, C, H, W)
        z_q_ste = z_q_ste.permute(0, 3, 1, 2).contiguous()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q_ste, z_q, indices