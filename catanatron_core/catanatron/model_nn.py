import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CatanResNet(nn.Module):
    def __init__(self, input_channels, H, W, action_size):
        super(CatanResNet, self).__init__()
        self.in_channels = 64
        self.H = H
        self.W = W

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Calculate the size of the flattened features
        self.flat_features = 512 * (H // 8) * (W // 8)
        
        # Layers for general action probabilities
        self.fc1 = nn.Linear(self.flat_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_action = nn.Linear(512, action_size)

        # Layers for spatial action probabilities
        self.conv_spatial1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv_spatial2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_spatial3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_spatial4 = nn.Conv2d(64, 1, kernel_size=1)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # General action probabilities
        flat = F.adaptive_avg_pool2d(out, (1, 1))
        flat = flat.view(flat.size(0), -1)
        flat = F.relu(self.fc1(flat))
        flat = F.relu(self.fc2(flat))
        action_probs = self.fc_action(flat)

        # Spatial action probabilities
        spatial = F.relu(self.conv_spatial1(out))
        spatial = F.relu(self.conv_spatial2(spatial))
        spatial = F.relu(self.conv_spatial3(spatial))
        spatial_probs = self.conv_spatial4(spatial)
        spatial_probs = spatial_probs.view(-1, self.H, self.W)

        return action_probs, spatial_probs

if __name__ == "__main__":
    input_channels = 37  # Adjust based on your actual input channels
    H, W = 15, 25  # Adjust based on your actual spatial dimensions
    output_size = 100  # Adjust based on your task (e.g., number of possible actions)

    model = CatanResNet(input_channels, H, W, output_size)