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
