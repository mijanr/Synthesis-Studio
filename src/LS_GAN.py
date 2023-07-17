import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generator class
    """
    def __init__(self, num_classes=10, latent_dim=100, img_shape=(1, 28, 28)):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes in dataset
        latent_dim : int
            Dimension of noise vector
        img_shape : tuple
            Shape of image
        """
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim + self.num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(torch.prod(torch.tensor(self.img_shape)))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        """
        Forward pass
        Parameters
        ----------
        noise : torch.Tensor
            Noise vector
        labels : torch.Tensor
            Labels of images

        Returns
        -------
        torch.Tensor
            Generated images
        """
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img
    
class Discriminator(nn.Module):
    """
    Discriminator class
    """
    def __init__(self, num_classes=10, img_shape=(1, 28, 28)):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes in dataset
        img_shape : tuple
            Shape of image
        """
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(self.num_classes + int(torch.prod(torch.tensor(self.img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, labels):
        """
        Forward pass
        Parameters
        ----------
        img : torch.Tensor
            Images
        labels : torch.Tensor
            Labels of images

        Returns
        -------
        torch.Tensor
            Probability of real image
        """
        img = img.view(img.size(0), -1)
        d_in = torch.cat((img, self.label_embedding(labels)), -1)
        return self.model(d_in)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator().to(device)
    num_classes = 10
    latent_dim = 100
    img_shape = (1, 28, 28)
    batch_size = 32
    noise = torch.randn(batch_size, latent_dim, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    gen_imgs = gen(noise, labels)
    print(gen_imgs.shape)
    disc = Discriminator().to(device)
    validity = disc(gen_imgs, labels)
    print(validity.shape)

