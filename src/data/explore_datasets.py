"""
Data Exploration Script for SONICS and GTZAN Datasets

This script:
1. Explores the structure of both datasets
2. Verifies file counts and formats
3. Samples audio files to check properties (sample rate, duration, channels)
4. Generates summary statistics
"""

import os
import glob
import random
from pathlib import Path
import numpy as np

# Dataset paths (from kagglehub cache)
SONICS_PATH = "/Users/stanleycliu/.cache/kagglehub/datasets/awsaf49/sonics-dataset/versions/2/fake_songs"
GTZAN_PATH = "/Users/stanleycliu/.cache/kagglehub/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/genres_original"


def explore_sonics():
    """Explore SONICS dataset (AI-generated songs)"""
    print("=" * 70)
    print("SONICS DATASET EXPLORATION (AI-Generated Songs)")
    print("=" * 70)

    # Get all .mp3 files
    ai_songs = glob.glob(os.path.join(SONICS_PATH, "*.mp3"))
    print(f"\nTotal AI song files (.mp3): {len(ai_songs)}")

    # Check file naming patterns
    print(f"\nSample filenames:")
    for song in random.sample(ai_songs, min(10, len(ai_songs))):
        print(f"  - {os.path.basename(song)}")

    # Check for 'suno' vs 'udio' generators
    suno_songs = [s for s in ai_songs if 'suno' in s]
    udio_songs = [s for s in ai_songs if 'udio' in s]
    print(f"\nAI Generator breakdown:")
    print(f"  - Suno songs: {len(suno_songs)}")
    print(f"  - Udio songs: {len(udio_songs)}")

    return ai_songs


def explore_gtzan():
    """Explore GTZAN dataset (Real songs)"""
    print("\n" + "=" * 70)
    print("GTZAN DATASET EXPLORATION (Real Songs)")
    print("=" * 70)

    # Get all genres
    genres = sorted([d for d in os.listdir(GTZAN_PATH)
                     if os.path.isdir(os.path.join(GTZAN_PATH, d))])
    print(f"\nGenres found: {len(genres)}")
    print(f"  {', '.join(genres)}")

    # Get all .wav files
    real_songs = []
    genre_counts = {}

    for genre in genres:
        genre_path = os.path.join(GTZAN_PATH, genre)
        genre_files = glob.glob(os.path.join(genre_path, "*.wav"))
        genre_counts[genre] = len(genre_files)
        real_songs.extend(genre_files)

    print(f"\nTotal real song files (.wav): {len(real_songs)}")
    print(f"\nSongs per genre:")
    for genre, count in sorted(genre_counts.items()):
        print(f"  - {genre:12s}: {count:3d} songs")

    # Sample filenames
    print(f"\nSample filenames:")
    for song in random.sample(real_songs, min(10, len(real_songs))):
        print(f"  - {os.path.basename(song)}")

    return real_songs


def check_audio_properties(ai_songs, real_songs, num_samples=5):
    """
    Check audio properties (sample rate, duration, channels)
    for a sample of files

    Note: Requires librosa to be installed
    """
    print("\n" + "=" * 70)
    print("AUDIO PROPERTIES CHECK")
    print("=" * 70)

    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("\n[INFO] librosa or soundfile not installed yet.")
        print("Run: pip install -r requirements.txt")
        print("Skipping audio properties check for now.")
        return

    print(f"\nChecking {num_samples} random AI songs (.mp3)...")
    for song_path in random.sample(ai_songs, min(num_samples, len(ai_songs))):
        try:
            audio, sr = librosa.load(song_path, sr=None)
            duration = len(audio) / sr
            print(f"\n  {os.path.basename(song_path)}")
            print(f"    - Sample rate: {sr} Hz")
            print(f"    - Duration: {duration:.2f} seconds")
            print(f"    - Channels: {'Stereo' if len(audio.shape) > 1 else 'Mono'}")
        except Exception as e:
            print(f"\n  {os.path.basename(song_path)}")
            print(f"    - Error loading: {e}")

    print(f"\nChecking {num_samples} random real songs (.wav)...")
    for song_path in random.sample(real_songs, min(num_samples, len(real_songs))):
        try:
            audio, sr = librosa.load(song_path, sr=None)
            duration = len(audio) / sr
            print(f"\n  {os.path.basename(song_path)}")
            print(f"    - Sample rate: {sr} Hz")
            print(f"    - Duration: {duration:.2f} seconds")
            print(f"    - Channels: {'Stereo' if len(audio.shape) > 1 else 'Mono'}")
        except Exception as e:
            print(f"\n  {os.path.basename(song_path)}")
            print(f"    - Error loading: {e}")


def generate_summary(ai_songs, real_songs):
    """Generate final summary"""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nDataset Statistics:")
    print(f"  - AI songs (SONICS):  {len(ai_songs):,} files (.mp3)")
    print(f"  - Real songs (GTZAN): {len(real_songs):,} files (.wav)")
    print(f"  - Total:              {len(ai_songs) + len(real_songs):,} files")

    print(f"\nFor balanced binary classification:")
    print(f"  - We will randomly sample 1,000 AI songs from SONICS")
    print(f"  - We will use all 1,000 real songs from GTZAN")
    print(f"  - Final dataset: 2,000 songs (perfectly balanced)")

    print(f"\nNext Steps:")
    print(f"  1. Install dependencies: pip install -r requirements.txt")
    print(f"  2. Run preprocessing to convert audio → mel spectrograms")
    print(f"  3. Create train/val/test splits (70/15/15)")
    print(f"  4. Train CNN classifier")

    print("\n" + "=" * 70)


def main():
    """Main exploration function"""
    print("\n")
    print("*" * 70)
    print("DATASET EXPLORATION FOR BINARY AUDIO CLASSIFIER")
    print("AI-Generated vs Real Music Classification")
    print("*" * 70)

    # Explore both datasets
    ai_songs = explore_sonics()
    real_songs = explore_gtzan()

    # Check audio properties (sample)
    check_audio_properties(ai_songs, real_songs, num_samples=3)

    # Generate summary
    generate_summary(ai_songs, real_songs)


if __name__ == "__main__":
    main()
