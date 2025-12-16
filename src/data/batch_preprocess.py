"""
Batch Preprocessing Script

This script:
1. Loads metadata from create_splits.py
2. Processes all songs: audio → mel spectrogram → save as .npy
3. Shows progress with tqdm
4. Saves preprocessed files to data/processed/
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

from audio_preprocessing import preprocess_and_save, get_spectrogram_shape


# Paths
METADATA_PATH = "data/metadata.csv"
OUTPUT_DIR_AI = "data/processed/ai_songs"
OUTPUT_DIR_REAL = "data/processed/real_songs"


def preprocess_all_songs(metadata_df: pd.DataFrame,
                         skip_existing: bool = True) -> dict:
    """
    Preprocess all songs in the metadata DataFrame.

    Args:
        metadata_df: DataFrame with file_path and label columns
        skip_existing: Skip files that have already been preprocessed

    Returns:
        Dictionary with processing statistics
    """
    total = len(metadata_df)
    successful = 0
    skipped = 0
    failed = 0
    failed_files = []

    print(f"\nProcessing {total:,} songs...")
    print(f"Output directories:")
    print(f"  AI songs:   {OUTPUT_DIR_AI}")
    print(f"  Real songs: {OUTPUT_DIR_REAL}")
    print()

    # Create output directories
    Path(OUTPUT_DIR_AI).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR_REAL).mkdir(parents=True, exist_ok=True)

    # Process each song
    for idx, row in tqdm(metadata_df.iterrows(), total=total, desc="Preprocessing"):
        audio_path = row['file_path']
        label = row['label']
        filename = row['filename']

        # Determine output directory and filename
        if label == 1:  # AI song
            output_dir = OUTPUT_DIR_AI
        else:  # Real song
            output_dir = OUTPUT_DIR_REAL

        # Create output filename (replace audio extension with .npy)
        output_filename = Path(filename).stem + '.npy'
        output_path = os.path.join(output_dir, output_filename)

        # Skip if already exists
        if skip_existing and os.path.exists(output_path):
            skipped += 1
            continue

        # Preprocess and save
        success = preprocess_and_save(audio_path, output_path, verbose=False)

        if success:
            successful += 1
        else:
            failed += 1
            failed_files.append(filename)

    # Return statistics
    stats = {
        'total': total,
        'successful': successful,
        'skipped': skipped,
        'failed': failed,
        'failed_files': failed_files
    }

    return stats


def update_metadata_with_npy_paths(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add npy_path column to metadata DataFrame.

    Args:
        metadata_df: Original metadata DataFrame

    Returns:
        Updated DataFrame with npy_path column
    """
    npy_paths = []

    for idx, row in metadata_df.iterrows():
        label = row['label']
        filename = row['filename']

        # Determine output directory
        if label == 1:  # AI song
            output_dir = OUTPUT_DIR_AI
        else:  # Real song
            output_dir = OUTPUT_DIR_REAL

        # Create .npy filename
        output_filename = Path(filename).stem + '.npy'
        npy_path = os.path.join(output_dir, output_filename)

        npy_paths.append(npy_path)

    # Add column
    metadata_df['npy_path'] = npy_paths

    return metadata_df


def print_statistics(stats: dict, start_time: float, end_time: float) -> None:
    """
    Print preprocessing statistics.

    Args:
        stats: Dictionary with processing statistics
        start_time: Start timestamp
        end_time: End timestamp
    """
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 70)
    print("PREPROCESSING STATISTICS")
    print("=" * 70)

    print(f"\nTotal songs:      {stats['total']:,}")
    print(f"Successful:       {stats['successful']:,}")
    print(f"Skipped:          {stats['skipped']:,}")
    print(f"Failed:           {stats['failed']:,}")

    if stats['failed'] > 0:
        print(f"\nFailed files:")
        for f in stats['failed_files'][:10]:  # Show first 10
            print(f"  - {f}")
        if len(stats['failed_files']) > 10:
            print(f"  ... and {len(stats['failed_files']) - 10} more")

    print(f"\nProcessing time:  {minutes}m {seconds}s")
    if stats['successful'] > 0:
        avg_time = elapsed / stats['successful']
        print(f"Avg per song:     {avg_time:.2f}s")

    print(f"\nExpected spectrogram shape: {get_spectrogram_shape()}")


def verify_random_samples(metadata_df: pd.DataFrame, num_samples: int = 5) -> None:
    """
    Verify random preprocessed samples.

    Args:
        metadata_df: Metadata DataFrame with npy_path column
        num_samples: Number of random samples to verify
    """
    import numpy as np
    import random

    print("\n" + "=" * 70)
    print("VERIFYING RANDOM SAMPLES")
    print("=" * 70)

    # Get samples with existing .npy files
    existing = metadata_df[metadata_df['npy_path'].apply(os.path.exists)]

    if len(existing) == 0:
        print("\nNo preprocessed files found to verify.")
        return

    samples = existing.sample(min(num_samples, len(existing)))

    expected_shape = get_spectrogram_shape()

    for idx, row in samples.iterrows():
        npy_path = row['npy_path']
        label_str = "AI" if row['label'] == 1 else "Real"

        spec = np.load(npy_path)

        print(f"\n{Path(npy_path).name} ({label_str})")
        print(f"  Shape: {spec.shape} (expected: {expected_shape})")
        print(f"  Mean:  {np.mean(spec):7.4f}")
        print(f"  Std:   {np.std(spec):7.4f}")
        print(f"  Min:   {np.min(spec):7.4f}")
        print(f"  Max:   {np.max(spec):7.4f}")

        if spec.shape == expected_shape:
            print(f"  ✅ Valid")
        else:
            print(f"  ❌ Invalid shape!")


def main():
    """Main preprocessing function"""
    print("\n" + "*" * 70)
    print("BATCH AUDIO PREPROCESSING")
    print("Converting Audio → Mel Spectrograms → .npy files")
    print("*" * 70)

    # Check if metadata exists
    if not os.path.exists(METADATA_PATH):
        print(f"\n❌ Metadata file not found: {METADATA_PATH}")
        print("Please run create_splits.py first!")
        return

    # Load metadata
    print(f"\nLoading metadata from: {METADATA_PATH}")
    metadata_df = pd.read_csv(METADATA_PATH)
    print(f"Loaded {len(metadata_df):,} songs")

    # Check dataset balance
    ai_count = (metadata_df['label'] == 1).sum()
    real_count = (metadata_df['label'] == 0).sum()
    print(f"  AI songs:   {ai_count:,}")
    print(f"  Real songs: {real_count:,}")

    # Confirm before processing
    print("\n" + "-" * 70)
    user_input = input("Start preprocessing? This may take 30-60 minutes. [y/N]: ")
    if user_input.lower() not in ['y', 'yes']:
        print("Preprocessing cancelled.")
        return

    # Start preprocessing
    start_time = time.time()

    stats = preprocess_all_songs(metadata_df, skip_existing=True)

    end_time = time.time()

    # Print statistics
    print_statistics(stats, start_time, end_time)

    # Update metadata with .npy paths
    print("\n" + "=" * 70)
    print("Updating metadata with .npy paths...")
    metadata_df = update_metadata_with_npy_paths(metadata_df)
    metadata_df.to_csv(METADATA_PATH, index=False)
    print(f"✅ Updated metadata saved to: {METADATA_PATH}")

    # Verify random samples
    verify_random_samples(metadata_df, num_samples=5)

    print("\n" + "=" * 70)
    print("✅ BATCH PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nPreprocessed spectrograms saved to:")
    print(f"  - {OUTPUT_DIR_AI}")
    print(f"  - {OUTPUT_DIR_REAL}")
    print(f"\nNext step: Train the CNN classifier!")


if __name__ == "__main__":
    main()
