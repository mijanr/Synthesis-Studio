from typing import Tuple
import torch
import torch.nn as nn

# Encoder class
class cEncoder(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int, num_classes:int)->torch.Tensor:
        super().__init__()
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input data, e.g. number of features (28*28=784 for MNIST)
        latent_dim : int
            Dimension of latent space (z)
        num_classes : int
            Number of classes in dataset, e.g. 10 for MNIST
        """
        self.embed = nn.Embedding(num_classes, num_classes)
        self.linear = nn.Sequential(
            nn.Linear(input_dim+num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.linear2 = nn.Linear(128, latent_dim)
        self.linear3 = nn.Linear(128, latent_dim)

    def forward(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data
        y : torch.Tensor
            Labels
        """
        y = self.embed(y)
        x = torch.cat((x, y), dim=1)
        hidden = self.linear(x)
        mu = self.linear2(hidden)
        log_var = self.linear3(hidden)
        return mu, log_var
    
