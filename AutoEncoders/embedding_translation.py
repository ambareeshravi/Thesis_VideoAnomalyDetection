import torch
from torch import nn
import numpy as np

class EmbeddingTranslator(nn.Module):
    def __init__(self, embedding_size = [1,128,2,2]):
        super(EmbeddingTranslator, self).__init__()
        embedding_size = np.product(embedding_size)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_size, embedding_size //2),
            nn.Tanh(),
            nn.Linear(embedding_size//2, embedding_size //4),
            nn.Tanh(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size//4, embedding_size//2),
            nn.Tanh(),
            nn.Linear(embedding_size//2, embedding_size),
            nn.Tanh(),
        )
        
    def forward(self, x):
        flat_x = x.flatten(start_dim=1, end_dim=-1)
        encodings = self.encoder(flat_x)
        recons = self.decoder(encodings)
        recons = recons.reshape(x.shape)
        return recons, encodings