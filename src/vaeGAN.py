"""
VAE-GAN model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        mu = self.fc31(h)
        logvar = self.fc32(h)
        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        x = self.sigmoid(self.fc3(h))
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        y = self.sigmoid(self.fc3(h))
        return y
    
class VAE_GAN(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(VAE_GAN, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.discriminator = Discriminator(input_dim, hidden_dim)
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        y = self.discriminator(x_hat)
        return x_hat, y, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def loss_function(self, x, x_hat, y, mu, logvar):
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Discriminator loss
        disc_loss = F.binary_cross_entropy(y, torch.ones_like(y), reduction='sum')
        # Generator loss
        gen_loss = F.binary_cross_entropy(y, torch.zeros_like(y), reduction='sum')
        # Total loss
        total_loss = recon_loss + kl_div + disc_loss + gen_loss
        return total_loss, recon_loss, kl_div, disc_loss, gen_loss
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        x_hat = self.decoder(z)
        return x_hat
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    input_dim = 784
    latent_dim = 100
    hidden_dim = 400
    output_dim = 784
    batch_size = 128
    num_samples = 16
    x = torch.randn(batch_size, input_dim).to(device)
    model = VAE_GAN(input_dim, latent_dim, hidden_dim, output_dim).to(device)
    x_hat, y, mu, logvar = model(x)
    print(x_hat.shape)
    print(y.shape)
    print(mu.shape)
    print(logvar.shape)
    total_loss, recon_loss, kl_div, disc_loss, gen_loss = model.loss_function(x, x_hat, y, mu, logvar)
    print(total_loss)
    print(recon_loss)
    print(kl_div)
    print(disc_loss)
    print(gen_loss)
    x_hat = model.sample(num_samples)
    print(x_hat.shape)