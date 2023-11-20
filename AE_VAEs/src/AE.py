import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, image_dim:int, latent_dim:int)->torch.Tensor:
        super(Encoder, self).__init__()
        """
        Encoder: E(x)
        Parameters:
            image_dim: dimension of image, e.g. 28*28=784
            latent_dim: dimension of latent vector z, e.g. 20
        Return:
            returns a tensor of latent vector z
        """
        self.seq = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        x: image tensor
        """
        return self.seq(x)
    
