
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        
        self.num_embeddings = num_embeddings  # codebook 크기 (K)
        self.embedding_dim = embedding_dim     # 벡터 차원 (D)
        self.commitment_cost = commitment_cost # β
        
        # Codebook 초기화
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z_e):
        """
        z_e: encoder 출력 (B, C, H, W)
        """
        # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_flat = z_e.view(-1, self.embedding_dim)
        
        # 각 z_e와 codebook 벡터 간의 거리 계산
        # (B*H*W, K)
        distances = (
            torch.sum(z_e_flat ** 2, dim=1, keepdim=True)  # (B*H*W, 1)
            - 2 * torch.matmul(z_e_flat, self.codebook.weight.t())  # (B*H*W, K)
            + torch.sum(self.codebook.weight ** 2, dim=1)  # (K,)
        )
        
        # 가장 가까운 codebook index 찾기
        indices = torch.argmin(distances, dim=1)  # (B*H*W,)
        
        # Codebook에서 벡터 가져오기
        z_q_flat = self.codebook(indices)  # (B*H*W, C)
        z_q = z_q_flat.view(z_e.shape)     # (B, H, W, C)
        
        # Loss 계산
        # VQ Loss: codebook 학습 (encoder에 stop gradient)
        vq_loss = F.mse_loss(z_q, z_e.detach())
        
        # Commitment Loss: encoder 학습 (codebook에 stop gradient)
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        
        loss = vq_loss + self.commitment_cost * commitment_loss
        
        # Straight-Through Estimator
        # Forward: z_q 사용, Backward: gradient를 z_e로 전달
        z_q = z_e + (z_q - z_e).detach()
        
        # (B, H, W, C) -> (B, C, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q, loss, indices