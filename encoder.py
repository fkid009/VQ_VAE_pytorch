import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            # (B, 3, 32, 32) -> (B, 64, 16, 16)
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # (B, 64, 16, 16) -> (B, 128, 8, 8)
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # (B, 128, 8, 8) -> (B, 128, 8, 8)
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # (B, 128, 8, 8) -> (B, embedding_dim, 8, 8)
            nn.Conv2d(hidden_channels * 2, embedding_dim, kernel_size=1, stride=1),
        )
    
    def forward(self, x):
        return self.net(x)