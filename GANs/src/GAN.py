"""
Author:
    Md Mijanur Rahman
"""
import torch
import torch.nn as nn

#Generator: G(z)
class Generator(nn.Module):
    def __init__(self, noise_dim: int, image_dim: int)->torch.Tensor:
        super(Generator, self).__init__()
        """
        args:
            noise_dim: dimension of noise vector, z, e.g. 100
            image_dim: dimension of image, x, e.g. 28*28 = 784 for MNIST
        return:
            return a tensor of image     
        """
        self.seq = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_dim),
            nn.Tanh()
        )
    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        x: noise vector
        """
        return self.seq(x)
    
#Discriminator: D(x)
class Discriminator(nn.Module):
    def __init__(self, image_dim: int)->torch.Tensor:
        super(Discriminator, self).__init__()
        """
        args:
            image_dim: dimension of image, x, e.g. 28*28 = 784 for MNIST
        return:
            return a tensor of image     
        """
        self.seq = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor)->torch.Tensor:   
        return self.seq(x)
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parameters
    image_dim = 28 * 28 * 1
    noise_dim = 100

    # check the generator and discriminator
    generator = Generator(noise_dim, image_dim).to(device)
    discriminator = Discriminator(image_dim).to(device)
    noise = torch.randn(64, noise_dim).to(device)
    gen_out = generator(noise)
    disc_out = discriminator(gen_out)
    print(gen_out.shape)
    print(disc_out.shape)




