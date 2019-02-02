import torch.nn as nn

NOISE = 100


class Generator(nn.Module):
    def __init__(self, noise=NOISE):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(noise, 64 * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64 * 4, 64 * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64 * 4, 64 * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16 * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16 * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16 * 2, 16 * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16 * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(16 * 4, 16 * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16 * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(16 * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        out = x.view(-1, 1).squeeze(1)
        return out
