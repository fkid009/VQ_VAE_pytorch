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
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(in_channels, hidden_channels, embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q_ste, z_q, indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q_ste)
        return x_recon, z_e, z_q, indices

    def encode(self, x):
        z_e = self.encoder(x)
        z_q_ste, _, indices = self.quantizer(z_e)
        return z_q_ste, indices

    def decode(self, z_q):
        return self.decoder(z_q)