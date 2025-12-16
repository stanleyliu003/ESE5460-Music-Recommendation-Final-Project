"""
CNN Architecture for Audio Classification (AI vs Real Music)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):
    """
    Convolutional Neural Network for binary audio classification

    Input shape: (batch_size, 1, 128, 938)
        - 1 channel (grayscale mel spectrogram)
        - 128 mel frequency bands (height)
        - 938 time frames (width)

    Output: (batch_size, 1) - probability of being AI-generated
    """

    def __init__(self, dropout_rate=0.5):
        super(AudioCNN, self).__init__()

        # Conv Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: (32, 64, 469)

        # Conv Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: (64, 32, 234)

        # Conv Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: (128, 16, 117)

        # Conv Block 4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # Output: (256, 8, 58)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (256, 1, 1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 1, 128, 938)

        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Global Average Pooling
        x = self.global_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

    def get_num_params(self):
        """Return the total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AudioCNNDeep(nn.Module):
    """
    Deeper CNN architecture with more convolutional layers

    For experimentation if the baseline model underfits
    """

    def __init__(self, dropout_rate=0.5):
        super(AudioCNNDeep, self).__init__()

        # Conv Block 1: 1 -> 32
        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv Block 2: 32 -> 64
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Conv Block 3: 64 -> 128
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Conv Block 4: 128 -> 256
        self.conv4a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool4(x)

        # Global pool and FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))

        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    """Test the model architecture"""

    # Test baseline model
    print("=" * 70)
    print("Testing AudioCNN (Baseline)")
    print("=" * 70)

    model = AudioCNN()
    print(f"\nModel architecture:\n{model}\n")
    print(f"Total trainable parameters: {model.get_num_params():,}")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 128, 938)
    print(f"\nInput shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output values (should be between 0 and 1):")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    print(f"  Mean: {output.mean().item():.4f}")

    # Test deep model
    print("\n" + "=" * 70)
    print("Testing AudioCNNDeep")
    print("=" * 70)

    deep_model = AudioCNNDeep()
    print(f"\nTotal trainable parameters: {deep_model.get_num_params():,}")

    output_deep = deep_model(dummy_input)
    print(f"\nOutput shape: {output_deep.shape}")
    print(f"Output values:")
    print(f"  Min: {output_deep.min().item():.4f}")
    print(f"  Max: {output_deep.max().item():.4f}")
    print(f"  Mean: {output_deep.mean().item():.4f}")

    print("\n" + "=" * 70)
    print("Model test completed successfully!")
    print("=" * 70)
