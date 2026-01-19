import torch   
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_channels, embedding_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            # (B, embedding_dim, 8, 8) -> (B, 128, 8, 8)
            nn.Conv2d(embedding_dim, hidden_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # (B, 128, 8, 8) -> (B, 64, 16, 16)
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # (B, 64, 16, 16) -> (B, 3, 32, 32)
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, z_q):
        return self.net(z_q)