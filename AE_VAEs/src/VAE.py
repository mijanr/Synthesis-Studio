import torch
import torch.nn as nn

# Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input data, e.g. number of features (28*28=784 for MNIST)
        hidden_dim : int
            Dimension of hidden layer
        latent_dim : int
            Dimension of latent space (z)
        """
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)
        
    
    def forward(self, x):
        hidden = self.linear1(x)
        mu = self.linear2(hidden)
        log_var = self.linear3(hidden)
        return mu, log_var
    
