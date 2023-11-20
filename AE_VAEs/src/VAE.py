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
    
# Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128, output_dim=784):
        super().__init__()
        """
        Parameters
        ----------
        latent_dim : int
            Dimension of latent space (z)
        hidden_dim : int
            Dimension of hidden layer
        output_dim : int    
            Dimension of output data, e.g. number of features (28*28=784 for MNIST)
        """
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.linear(x)
    
# VAE class
class VAE(nn.Module):
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
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        output = self.decoder(z)
    
    def reparameterize(self, mu, log_var):
        """
        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent Gaussian distribution
        log_var : torch.Tensor
            Log variance of the latent Gaussian distribution
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
    
    def loss_function(self, x, output, mu, log_var):
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
        """
        # Reconstruction loss
        # MSE loss
        mse_loss = nn.MSELoss(reduction='sum')
        recon_loss = mse_loss(output, x)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Total loss
        total_loss = recon_loss + kl_loss
        return total_loss
    
    def sample(self, num_samples):
        """
        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z = torch.randn(num_samples, self.encoder.linear2.out_features)
        samples = self.decoder(z.to(device))
        return samples
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # AE parameters
    input_dim = 784
    hidden_dim = 400
    latent_dim = 20
    ae = VAE(input_dim, hidden_dim, latent_dim).to(device)

    # test forward pass
    x = torch.randn(64, input_dim).to(device)
    output, mu, log_var = ae(x)
    print('output.shape:', output.shape)
    print('mu.shape:', mu.shape)
    print('log_var.shape:', log_var.shape)

    # random sample
    samples = ae.sample(num_samples=64)
    print('decoder_out.shape:', samples.shape)

    # test loss function
    loss = ae.loss_function(x, output, mu, log_var)
    print('loss:', loss.item())

