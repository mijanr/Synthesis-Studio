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
    
class Decoder(nn.Module):
    def __init__(self, image_dim:int, latent_dim:int)->torch.Tensor:
        super(Decoder, self).__init__()
        """
        Decoder: D(z)
        Parameters:
            image_dim: dimension of image, e.g. 28*28=784
            latent_dim: dimension of latent vector z, e.g. 20
        Return:
            returns a tensor of image
        """
        self.seq = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_dim),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        x: latent vector z
        """
        return self.seq(x)
    
class AE(nn.Module):
    def __init__(self, image_dim:int, latent_dim:int)->torch.Tensor:
        super(AE, self).__init__()
        """
        Autoencoder: AE(x)
        Parameters:
            image_dim: dimension of image, e.g. 28*28=784
            latent_dim: dimension of latent vector z, e.g. 20
        Return:
            returns a tuple of two tensors:
                - first tensor: reconstructed image
                - second tensor: encoded image
        """
        self.image_dim = image_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(image_dim, latent_dim)
        self.decoder = Decoder(image_dim, latent_dim)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        x: image tensor
        """
        encoded_img = self.encoder(x)
        reconstructed_img = self.decoder(encoded_img)
        assert encoded_img.shape[-1] == self.latent_dim
        assert reconstructed_img.shape == x.shape
        return reconstructed_img, encoded_img
    
if __name__ == "__main__":
    # create an instance of AE
    image_dim = 784
    latent_dim = 20
    model = AE(image_dim, latent_dim)

    # create a random image tensor
    x = torch.rand(1, 784)

    # forward pass
    reconstructed_img, encoded_img = model(x)
    print(reconstructed_img.shape)
    print(encoded_img.shape)
