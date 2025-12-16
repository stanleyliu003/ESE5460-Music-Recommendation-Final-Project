"""
Main training script for Audio Classifier
Run this to train the model for 100 epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "src"))

from data.dataset import create_dataloaders
from models.cnn_classifier import AudioCNN
from models.train import Trainer


def main():
    """Main training function"""

    print("=" * 70)
    print("AUDIO CLASSIFIER TRAINING - 20 EPOCHS")
    print("=" * 70)

    # Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = None  # No early stopping
    NUM_WORKERS = 4

    # Paths
    project_root = Path(__file__).parent
    train_csv = project_root / "data" / "splits" / "train.csv"
    val_csv = project_root / "data" / "splits" / "val.csv"
    test_csv = project_root / "data" / "splits" / "test.csv"
    checkpoint_dir = project_root / "checkpoints"
    results_dir = project_root / "results"

    # Create directories
    checkpoint_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv, val_csv, test_csv,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print(f"✅ Train samples: {len(train_loader.dataset)}")
    print(f"✅ Val samples: {len(val_loader.dataset)}")
    print(f"✅ Test samples: {len(test_loader.dataset)}")
    print()

    # Create model
    model = AudioCNN(dropout_rate=0.5)
    print(f"Model: AudioCNN")
    print(f"Total parameters: {model.get_num_params():,}")
    print()

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        model_name="audio_cnn_20epochs"
    )

    # Train
    print("Starting training...")
    print()
    history = trainer.train(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE if EARLY_STOPPING_PATIENCE else 999
    )

    print("\n✅ Training complete!")
    print(f"📊 Results saved to: {checkpoint_dir}")
    print(f"📈 To visualize: python src/utils/visualization.py {checkpoint_dir}/audio_cnn_20epochs_history.json")


if __name__ == "__main__":
    main()
