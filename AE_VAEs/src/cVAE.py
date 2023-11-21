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
    
# Decoder class
class cDecoder(nn.Module):
    def __init__(self, latent_dim:int, output_dim:int, num_classes:int)->torch.Tensor:
        super().__init__()
        """
        Parameters
        ----------
        latent_dim : int
            Dimension of latent space (z)
        output_dim : int
            Dimension of output data, e.g. number of features (28*28=784 for MNIST)
        num_classes : int
            Number of classes in dataset, e.g. 10 for MNIST
        """
        self.embed = nn.Embedding(num_classes, num_classes)
        self.linear = nn.Sequential(
            nn.Linear(latent_dim+num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor
            Latent space
        y : torch.Tensor
            Labels
        """
        y = self.embed(y)
        z = torch.cat((z, y), dim=1)
        return self.linear(z)

# cVAE class
class cVAE(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int, output_dim:int, num_classes:int)->torch.Tensor:
        super().__init__()
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input data, e.g. number of features (28*28=784 for MNIST)
        latent_dim : int
            Dimension of latent space (z)
        output_dim : int
            Dimension of output data, e.g. number of features (28*28=784 for MNIST)
        num_classes : int
            Number of classes in dataset, e.g. 10 for MNIST
        """
        self.latent_dim = latent_dim
        self.encoder = cEncoder(input_dim, latent_dim, num_classes)
        self.decoder = cDecoder(latent_dim, output_dim, num_classes)

    def forward(self, x:torch.Tensor, y:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data
        y : torch.Tensor
            Labels
        
        Returns
        -------
        reconstructed_x : torch.Tensor
            Reconstructed input data
        mu : torch.Tensor
            Mean of latent space
        log_var : torch.Tensor
            Log variance of latent space
        """
        mu : torch.Tensor
        mu, log_var = self.encoder(x, y)
        z = self.reparameterize(mu, log_var)
        output = self.decoder(z, y)
        return output, mu, log_var
    
    def reparameterize(self, mu:torch.Tensor, log_var:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent Gaussian distribution
        log_var : torch.Tensor
            Log variance of the latent Gaussian distribution
        
        Returns
        -------
        z : torch.Tensor
            Latent space
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
    
    def loss_function(self, x:torch.Tensor, output:torch.Tensor, mu:torch.Tensor, log_var:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data
        output : torch.Tensor
            Output data
        mu : torch.Tensor
            Mean of the latent Gaussian distribution
        log_var : torch.Tensor
            Log variance of the latent Gaussian distribution
        
        Returns
        -------
        loss : torch.Tensor
            Loss value
        """
        # MSE loss
        mse_loss = nn.MSELoss(reduction='sum')
        # Reconstruction loss
        recon_loss = mse_loss(output, x)
        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Total loss
        loss = recon_loss + KLD
        return loss
    
    def sample(self, num_samples:int, y:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        y : torch.Tensor
            Labels
        
        Returns
        -------
        samples : torch.Tensor
            Generated samples
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z, y)
        return samples


