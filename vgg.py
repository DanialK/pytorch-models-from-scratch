from typing import List, Union, cast
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class VGG(nn.Module):
    def __init__(self, layers_config: List[Union[int, str]], num_classes: int, in_channels: int = 3,
                 dropout: float = 0.5):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.make_conv_layers(layers_config)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        x = x.reshape(-1)
        x = self.classifier(x)
        return x

    def make_conv_layers(self, layers_config: List[Union[int, str]]) -> nn.Module:
        layers: List[nn.Module] = []

        in_channels = self.in_channels

        for layer in layers_config:
            if layer == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                layer = cast(int, layer)
                layers += [
                    nn.Conv2d(in_channels, layer, kernel_size=3, padding=1),
                    nn.BatchNorm2d(layer),
                    nn.ReLU()
                ]
                in_channels = layer

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG(VGG16, num_classes=10, in_channels=3).to(device)
    x = torch.randn(1, 3, 224, 224).float().to(device)
    print(model(x))
