import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from quantizer import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=64,
        embedding_dim=64,
        num_embeddings=512,
        commitment_cost=0.25
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(in_channels, hidden_channels, embedding_dim)
    
    def forward(self, x):
        # Encode
        z_e = self.encoder(x)
        
        # Quantize
        z_q, vq_loss, indices = self.quantizer(z_e)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss, indices
    
    def encode(self, x):
        """인코딩만 수행 (inference용)"""
        z_e = self.encoder(x)
        z_q, _, indices = self.quantizer(z_e)
        return z_q, indices
    
    def decode(self, z_q):
        """디코딩만 수행 (inference용)"""
        return self.decoder(z_q)