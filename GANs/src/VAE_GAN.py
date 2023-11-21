"""
VAE-GAN model
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int, hidden_dim:int)->Tuple[torch.Tensor, torch.Tensor]:
        super(Encoder, self).__init__()
        """
        Encoder network

        Parameters
        ----------
        input_dim : int
            input dimension, e.g. 28*28=784 for MNIST dataset
        latent_dim : int
            dimension of the latent variable z
        hidden_dim : int
            dimension of the hidden layer
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            mean and log variance of the latent variable z
        """
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        mu = self.fc31(h)
        logvar = self.fc32(h)
        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim:int, hidden_dim:int, output_dim:int)->torch.Tensor:
        super(Decoder, self).__init__()
        """
        Decoder network

        Parameters
        ----------
        latent_dim : int
            dimension of the latent variable z  
        hidden_dim : int
            dimension of the hidden layer
        output_dim : int
            output dimension, e.g. 28*28=784 for MNIST dataset

        Returns
        ------- 
        torch.Tensor
            reconstructed data x_hat 
        """
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, z:torch.Tensor)->torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        z : torch.Tensor
            latent variable z

        Returns
        -------
        torch.Tensor
            reconstructed data x_hat
        """
        return self.fc(z)
    
class Discriminator(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int)->torch.Tensor:
        super(Discriminator, self).__init__()
        """
        Discriminator network

        Parameters
        ----------
        input_dim : int
            input dimension, e.g. 28*28=784 for MNIST dataset
        hidden_dim : int
            dimension of the hidden layer

        Returns
        -------
        torch.Tensor
            probability of the input data being real
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            probability of the input data being real
        """
        return self.fc(x)
    
class VAE_GAN(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int, hidden_dim:int, output_dim:int)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        super(VAE_GAN, self).__init__()
        """
        VAE-GAN model

        Parameters
        ----------
        input_dim : int
            input dimension, e.g. 28*28=784 for MNIST dataset
        latent_dim : int
            dimension of the latent variable z
        hidden_dim : int
            dimension of the hidden layer
        output_dim : int    
            output dimension, e.g. 28*28=784 for MNIST dataset

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            reconstructed data x_hat, probability of the input data being real, mean and log variance of the latent variable z
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.discriminator = Discriminator(input_dim, hidden_dim)
        
    def forward(self, x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            reconstructed data x_hat, probability of the input data being real, mean and log variance of the latent variable z
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        y = self.discriminator(x_hat)
        return x_hat, y, mu, logvar
    
    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor)->torch.Tensor:
        """
        Reparameterization trick

        Parameters
        ----------
        mu : torch.Tensor
            mean of the latent variable z
        logvar : torch.Tensor
            log variance of the latent variable z

        Returns
        -------
        torch.Tensor
            latent variable z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def loss_function(self, x:torch.Tensor, x_hat:torch.Tensor, y:torch.Tensor, mu:torch.Tensor, logvar:torch.Tensor):
        """
        Loss function

        Parameters
        ----------
        x : torch.Tensor
            input data
        x_hat : torch.Tensor
            reconstructed data x_hat
        y : torch.Tensor
            probability of the input data being real
        mu : torch.Tensor
            mean of the latent variable z
        logvar : torch.Tensor
            log variance of the latent variable z

        Returns
        -------
        total_loss : torch.Tensor
            total loss
        recon_loss : torch.Tensor
            reconstruction loss
        kl_div : torch.Tensor   
            KL divergence
        disc_loss : torch.Tensor
            discriminator loss
        gen_loss : torch.Tensor
            generator loss
        """
        # Reconstruction loss
        mse_loss = nn.MSELoss(reduction='sum')
        bce_loss = nn.BCELoss(reduction='sum')
        recon_loss = mse_loss(x_hat, x)
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Discriminator loss
        disc_loss = bce_loss(y, torch.ones_like(y))
        # Generator loss
        gen_loss = bce_loss(y, torch.zeros_like(y))
        # Total loss
        total_loss = recon_loss + kl_div + disc_loss + gen_loss
        return total_loss#, recon_loss, kl_div, disc_loss, gen_loss
    
    def sample(self, num_samples:int)->torch.Tensor:
        """
        Sample from the latent space

        Parameters
        ----------
        num_samples : int
            number of samples

        Returns
        -------
        torch.Tensor
            samples from the latent space
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z = torch.randn(num_samples, self.latent_dim).to(device)
        x_hat = self.decoder(z)
        return x_hat
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parameters
    input_dim = 784
    latent_dim = 100
    hidden_dim = 400
    output_dim = 784
    batch_size = 128
    num_samples = 16

    # test vae-gan
    x = torch.randn(batch_size, input_dim).to(device)
    model = VAE_GAN(input_dim, latent_dim, hidden_dim, output_dim).to(device)
    x_hat, y, mu, logvar = model(x)
    print(x_hat.shape)
    print(y.shape)
    print(mu.shape)
    print(logvar.shape)

    # test loss function
    total_loss = model.loss_function(x, x_hat, y, mu, logvar)
    print(total_loss)

    # test sample
    x_hat = model.sample(num_samples)
    print(x_hat.shape)