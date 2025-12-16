"""
Training script for Audio CNN Classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import AudioSpectrogramDataset, create_dataloaders
from models.cnn_classifier import AudioCNN, AudioCNNDeep


class Trainer:
    """Training class for Audio CNN"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        checkpoint_dir="checkpoints",
        model_name="audio_cnn"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for specs, labels in pbar:
            specs = specs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(specs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * specs.size(0)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]  ")
            for specs, labels in pbar:
                specs = specs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * specs.size(0)
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pth"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model to {best_path}")

    def train(self, num_epochs, early_stopping_patience=10):
        """
        Train the model

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if val loss doesn't improve for this many epochs
        """
        print("=" * 70)
        print(f"Starting Training: {self.model_name}")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Epochs: {num_epochs}")
        print("=" * 70)

        patience_counter = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate(epoch)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                patience_counter = 0
                print(f"  🎯 New best validation loss!")
            else:
                patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                print(f"   Val loss hasn't improved for {early_stopping_patience} epochs")
                break

            print("-" * 70)

        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print("=" * 70)

        # Save training history
        history_path = self.checkpoint_dir / f"{self.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {history_path}")

        return self.history


def main():
    """Main training function"""

    # Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    NUM_WORKERS = 4
    MODEL_TYPE = "baseline"  # "baseline" or "deep"

    # Paths
    project_root = Path(__file__).parent.parent.parent
    train_csv = project_root / "data" / "splits" / "train.csv"
    val_csv = project_root / "data" / "splits" / "val.csv"
    test_csv = project_root / "data" / "splits" / "test.csv"
    checkpoint_dir = project_root / "checkpoints"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv, val_csv, test_csv,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    if MODEL_TYPE == "deep":
        model = AudioCNNDeep()
        model_name = "audio_cnn_deep"
    else:
        model = AudioCNN()
        model_name = "audio_cnn"

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
        model_name=model_name
    )

    # Train
    history = trainer.train(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )

    print("\nTraining complete! Check the checkpoints directory for saved models.")


if __name__ == "__main__":
    main()
