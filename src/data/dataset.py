"""
PyTorch Dataset class for audio mel spectrograms
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


class AudioSpectrogramDataset(Dataset):
    """
    Dataset for loading preprocessed mel spectrograms from .npy files

    Args:
        csv_path: Path to CSV file containing file paths and labels
        transform: Optional transform to apply to spectrograms
    """

    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        # Verify all files exist
        missing_files = []
        for idx, row in self.data.iterrows():
            if not Path(row['npy_path']).exists():
                missing_files.append(row['npy_path'])

        if missing_files:
            print(f"Warning: {len(missing_files)} files not found")
            print(f"First missing file: {missing_files[0]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            spec: Tensor of shape (1, 128, 938) - (channels, mel_bands, time_frames)
            label: Tensor of shape (1,) - binary label (0=real, 1=AI)
        """
        row = self.data.iloc[idx]

        # Load preprocessed mel spectrogram
        spec = np.load(row['npy_path'])  # Shape: (128, 938)
        label = row['label']  # 0 or 1

        # Add channel dimension: (128, 938) -> (1, 128, 938)
        spec = torch.FloatTensor(spec).unsqueeze(0)

        # Apply transforms if provided
        if self.transform:
            spec = self.transform(spec)

        # Convert label to tensor
        label = torch.FloatTensor([label])

        return spec, label


def get_spectrogram_shape():
    """
    Returns the expected shape of mel spectrograms
    """
    return (1, 128, 938)  # (channels, mel_bands, time_frames)


def create_dataloaders(train_csv, val_csv, test_csv, batch_size=32,
                       num_workers=4, transform=None):
    """
    Create DataLoaders for train, validation, and test sets

    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        transform: Optional transform to apply

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = AudioSpectrogramDataset(train_csv, transform=transform)
    val_dataset = AudioSpectrogramDataset(val_csv, transform=transform)
    test_dataset = AudioSpectrogramDataset(test_csv, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test the dataset"""
    import sys
    from pathlib import Path

    # Test with sample data
    train_csv = Path(__file__).parent.parent.parent / "data" / "splits" / "train.csv"

    if train_csv.exists():
        print(f"Loading dataset from: {train_csv}")
        dataset = AudioSpectrogramDataset(train_csv)

        print(f"\nDataset size: {len(dataset)}")

        # Get first sample
        spec, label = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Spectrogram shape: {spec.shape}")
        print(f"  Label: {label.item()} ({'AI' if label.item() == 1 else 'Real'})")
        print(f"  Spectrogram stats:")
        print(f"    Mean: {spec.mean().item():.4f}")
        print(f"    Std: {spec.std().item():.4f}")
        print(f"    Min: {spec.min().item():.4f}")
        print(f"    Max: {spec.max().item():.4f}")

        # Check class balance
        labels = [dataset[i][1].item() for i in range(min(100, len(dataset)))]
        ai_count = sum(labels)
        real_count = len(labels) - ai_count
        print(f"\nClass balance (first 100 samples):")
        print(f"  Real songs: {real_count}")
        print(f"  AI songs: {ai_count}")

    else:
        print(f"Error: {train_csv} not found")
        print("Please run create_splits.py first")
