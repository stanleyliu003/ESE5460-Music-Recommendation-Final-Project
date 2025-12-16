"""
Visualization utilities for training and evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path


sns.set_style("whitegrid")


def plot_training_history(history_path, save_path=None):
    """
    Plot training and validation loss/accuracy curves

    Args:
        history_path: Path to training history JSON file
        save_path: Optional path to save the plot
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Total epochs: {len(epochs)}")
    print(f"\nFinal Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"\nFinal Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"\nBest Validation Loss: {min(history['val_loss']):.4f} (Epoch {history['val_loss'].index(min(history['val_loss'])) + 1})")
    print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}% (Epoch {history['val_acc'].index(max(history['val_acc'])) + 1})")
    print("=" * 70)


def plot_confusion_matrix(cm, class_names=['Real', 'AI'], save_path=None):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix (2x2 numpy array)
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_learning_rate(history_path, save_path=None):
    """
    Plot learning rate schedule

    Args:
        history_path: Path to training history JSON file
        save_path: Optional path to save the plot
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    if 'learning_rates' not in history or not history['learning_rates']:
        print("No learning rate data found in history")
        return

    epochs = range(1, len(history['learning_rates']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    """Test visualization with example data"""
    import sys

    # Check if history file is provided
    if len(sys.argv) > 1:
        history_path = sys.argv[1]
        if Path(history_path).exists():
            print(f"Loading training history from: {history_path}")
            plot_training_history(history_path)
            plot_learning_rate(history_path)
        else:
            print(f"Error: {history_path} not found")
    else:
        print("Usage: python visualization.py <path_to_history.json>")
        print("Example: python visualization.py ../../checkpoints/audio_cnn_history.json")
