"""
Author: Md Mijanur Rahman
"""
import torch
import torch.nn as nn


# conditional generator
class cGenerator(nn.Module):
    def __init__(self, noise_dim:int, n_classes:int):
        super(cGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.n_classes = n_classes
        self.embed = nn.Embedding(n_classes, n_classes)
        self.linear = nn.Linear(noise_dim+n_classes, 128*7*7)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, labels):
        embed = self.embed(labels)
        x = torch.cat([x, embed], dim=1)
        x = self.linear(x)
        x = x.view(-1, 128, 7, 7)
        x = self.conv(x)
        return x

# conditional discriminator
class cDiscriminator(nn.Module):
    def __init__(self, n_classes:int):
        super(cDiscriminator, self).__init__()
        self.n_classes = n_classes
        self.embed = nn.Embedding(n_classes, n_classes)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features= 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(-1, 28, 28)
        labels = self.embed(labels)
        labels = labels.unsqueeze(2).repeat(1, 1, 28)
        x = torch.cat([x, labels], dim=1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.linear(x)
        return x    

