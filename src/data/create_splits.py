"""
Create Train/Validation/Test Splits

This script:
1. Randomly samples 1,000 AI songs from SONICS dataset
2. Uses all 1,000 real songs from GTZAN dataset
3. Creates stratified 70/15/15 split for train/val/test
4. Saves file paths and labels to CSV
5. Saves split indices to JSON
"""

import os
import json
import random
import glob
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict


# Dataset paths
SONICS_PATH = "/Users/stanleycliu/.cache/kagglehub/datasets/awsaf49/sonics-dataset/versions/2/fake_songs"
GTZAN_PATH = "/Users/stanleycliu/.cache/kagglehub/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/genres_original"

# Output paths
OUTPUT_DIR = "data/splits"
METADATA_PATH = "data/metadata.csv"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42


def get_ai_songs(num_samples: int = 1000) -> List[str]:
    """
    Randomly sample AI-generated songs from SONICS dataset.

    Args:
        num_samples: Number of AI songs to sample (default: 1000)

    Returns:
        List of file paths to AI songs
    """
    # Get all AI songs (.mp3 files)
    all_ai_songs = glob.glob(os.path.join(SONICS_PATH, "*.mp3"))

    print(f"Total AI songs available: {len(all_ai_songs):,}")

    # Randomly sample
    random.seed(RANDOM_SEED)
    sampled_ai_songs = random.sample(all_ai_songs, num_samples)

    print(f"Sampled AI songs: {len(sampled_ai_songs):,}")

    return sampled_ai_songs


def get_real_songs() -> List[str]:
    """
    Get all real songs from GTZAN dataset.

    Returns:
        List of file paths to real songs
    """
    # Get all genres
    genres = sorted([d for d in os.listdir(GTZAN_PATH)
                     if os.path.isdir(os.path.join(GTZAN_PATH, d))])

    # Get all .wav files from all genres
    real_songs = []
    for genre in genres:
        genre_path = os.path.join(GTZAN_PATH, genre)
        genre_files = glob.glob(os.path.join(genre_path, "*.wav"))
        real_songs.extend(genre_files)

    print(f"Total real songs: {len(real_songs):,}")

    return real_songs


def create_dataset_metadata(ai_songs: List[str],
                            real_songs: List[str]) -> pd.DataFrame:
    """
    Create metadata DataFrame with file paths and labels.

    Args:
        ai_songs: List of AI song file paths
        real_songs: List of real song file paths

    Returns:
        DataFrame with columns: file_path, label, source
    """
    # Create lists for DataFrame
    file_paths = []
    labels = []
    sources = []

    # Add AI songs (label = 1)
    for song in ai_songs:
        file_paths.append(song)
        labels.append(1)  # 1 = AI-generated
        sources.append('SONICS')

    # Add real songs (label = 0)
    for song in real_songs:
        file_paths.append(song)
        labels.append(0)  # 0 = Real
        sources.append('GTZAN')

    # Create DataFrame
    df = pd.DataFrame({
        'file_path': file_paths,
        'label': labels,
        'source': sources,
        'filename': [os.path.basename(fp) for fp in file_paths]
    })

    return df


def create_stratified_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.

    Args:
        df: Metadata DataFrame

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # First split: train + val (85%) vs test (15%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df['label'],
        random_state=RANDOM_SEED
    )

    # Second split: train (70% of total) vs val (15% of total)
    # Since train_val is 85% of total, we split it as 70/85 vs 15/85
    train_ratio_adjusted = TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=(1 - train_ratio_adjusted),
        stratify=train_val_df['label'],
        random_state=RANDOM_SEED
    )

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                test_df: pd.DataFrame,
                output_dir: str = OUTPUT_DIR) -> None:
    """
    Save split information to files.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save split files
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Add split column
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Combine all splits
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Save metadata CSV
    metadata_path = METADATA_PATH
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(metadata_path, index=False)
    print(f"\n✅ Saved metadata to: {metadata_path}")

    # Save split indices as JSON
    split_info = {
        'train_indices': train_df.index.tolist(),
        'val_indices': val_df.index.tolist(),
        'test_indices': test_df.index.tolist(),
        'random_seed': RANDOM_SEED
    }

    split_json_path = os.path.join(output_dir, 'split_indices.json')
    with open(split_json_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"✅ Saved split indices to: {split_json_path}")

    # Save individual split CSVs
    train_csv = os.path.join(output_dir, 'train.csv')
    val_csv = os.path.join(output_dir, 'val.csv')
    test_csv = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"✅ Saved train split to: {train_csv}")
    print(f"✅ Saved val split to: {val_csv}")
    print(f"✅ Saved test split to: {test_csv}")


def print_split_statistics(train_df: pd.DataFrame,
                           val_df: pd.DataFrame,
                           test_df: pd.DataFrame) -> None:
    """
    Print statistics for each split.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
    """
    print("\n" + "=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)

    for name, df in [('Training', train_df), ('Validation', val_df), ('Test', test_df)]:
        total = len(df)
        ai_count = (df['label'] == 1).sum()
        real_count = (df['label'] == 0).sum()
        ai_pct = (ai_count / total) * 100
        real_pct = (real_count / total) * 100

        print(f"\n{name} Set:")
        print(f"  Total: {total:4d} songs")
        print(f"  AI:    {ai_count:4d} songs ({ai_pct:.1f}%)")
        print(f"  Real:  {real_count:4d} songs ({real_pct:.1f}%)")

    # Overall statistics
    total_songs = len(train_df) + len(val_df) + len(test_df)
    print(f"\nOverall Total: {total_songs:,} songs")
    print(f"Split Ratio: {len(train_df)/total_songs:.0%} / "
          f"{len(val_df)/total_songs:.0%} / "
          f"{len(test_df)/total_songs:.0%}")


def main():
    """Main function to create splits"""
    print("\n" + "*" * 70)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("*" * 70)

    # Get AI songs (sample 1000)
    print("\n[1/5] Sampling AI-generated songs from SONICS...")
    ai_songs = get_ai_songs(num_samples=1000)

    # Get real songs (all 1000)
    print("\n[2/5] Loading real songs from GTZAN...")
    real_songs = get_real_songs()

    # Create metadata
    print("\n[3/5] Creating metadata DataFrame...")
    df = create_dataset_metadata(ai_songs, real_songs)
    print(f"Total dataset size: {len(df):,} songs")

    # Create splits
    print("\n[4/5] Creating stratified train/val/test splits...")
    train_df, val_df, test_df = create_stratified_splits(df)

    # Print statistics
    print_split_statistics(train_df, val_df, test_df)

    # Save splits
    print("\n[5/5] Saving split information...")
    save_splits(train_df, val_df, test_df)

    print("\n" + "=" * 70)
    print("✅ SPLIT CREATION COMPLETE!")
    print("=" * 70)
    print(f"\nNext step: Run preprocessing to convert audio → spectrograms")


if __name__ == "__main__":
    main()
