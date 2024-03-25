import torch
import torch.nn as nn


#Generator: G(z, y)
class Generator(nn.Module):
    def __init__(self, noise_dim: int, image_dim: int, num_classes:int)->torch.Tensor:
        """
        Parameters
        ----------
        noise_dim : int
            dimension of noise vector
        image_dim : int
            dimension of image e.g. 784 for MNIST
        num_classes : int
            number of classes e.g. 10 for MNIST

        Returns
        -------
        forward : torch.Tensor
            return a tensor of image
        """

        super(Generator, self).__init__()
        self.embed = nn.Embedding(num_classes, num_classes)
        self.seq = nn.Sequential(
            nn.Linear(noise_dim+num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
        """
        x: noise vector
        y: label
        """
        y = self.embed(y)
        x = torch.cat([x, y], dim=1)
        return self.seq(x)
    
#Discriminator: D(x, y)
class Discriminator(nn.Module):
    def __init__(self, image_dim: int, num_classes:int)->torch.Tensor:
        """
        Parameters
        ----------
        image_dim : int
            dimension of image e.g. 784 for MNIST
        num_classes : int
            number of classes e.g. 10 for MNIST

        Returns
        -------
        forward : torch.Tensor
            return a tensor of image, shape (batch_size, 1)
        """
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(num_classes, num_classes)
        self.seq = nn.Sequential(
            nn.Linear(image_dim+num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
        """
        x: image
        y: label
        """
        y = self.embed(y)
        x = torch.cat([x, y], dim=1)
        return self.seq(x)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_dim = 100
    image_dim = 784
    num_classes = 10
    batch_size = 128
    noise = torch.randn(batch_size, noise_dim).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    G = Generator(noise_dim, image_dim, num_classes).to(device)
    D = Discriminator(image_dim, num_classes).to(device)
    gen_out = G(noise, labels)
    dis_out = D(gen_out, labels)
    print(gen_out.shape)
    print(dis_out.shape)
    
