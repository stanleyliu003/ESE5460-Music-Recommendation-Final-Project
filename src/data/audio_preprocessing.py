"""
Audio Preprocessing Module

Handles:
1. Loading audio files (.mp3 and .wav)
2. Resampling to target sample rate (16,000 Hz - downsampling, no upsampling)
3. Trimming/padding to exact duration (30 seconds)
4. Converting to mel spectrogram
5. Normalization
6. Saving as .npy files

Note: SONICS is 16,000 Hz, GTZAN is 22,050 Hz.
We downsample to 16,000 Hz to avoid upsampling artifacts.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional


# Preprocessing configuration
TARGET_SR = 16000  # Target sample rate (Hz) - use SONICS rate, downsample GTZAN
DURATION = 30.0    # Duration in seconds
N_MELS = 128       # Number of mel bands
N_FFT = 2048       # FFT window size
HOP_LENGTH = 512   # Hop length for STFT


def load_audio(file_path: str,
               sample_rate: int = TARGET_SR,
               duration: float = DURATION) -> np.ndarray:
    """
    Load audio file and prepare it for processing.

    Args:
        file_path: Path to audio file (.mp3 or .wav)
        sample_rate: Target sample rate (default: 22050 Hz)
        duration: Target duration in seconds (default: 30.0)

    Returns:
        Audio waveform as numpy array
    """
    # Load audio with librosa (handles both .mp3 and .wav)
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)

    # Calculate target length in samples
    target_length = int(sample_rate * duration)

    # Trim or pad to exact length
    if len(audio) > target_length:
        # Trim: take first 30 seconds
        audio = audio[:target_length]
    elif len(audio) < target_length:
        # Pad with zeros at the end
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')

    return audio


def audio_to_mel_spectrogram(audio: np.ndarray,
                              sample_rate: int = TARGET_SR,
                              n_mels: int = N_MELS,
                              n_fft: int = N_FFT,
                              hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Convert audio waveform to mel spectrogram in dB scale.

    Args:
        audio: Audio waveform (1D numpy array)
        sample_rate: Sample rate of audio
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT

    Returns:
        Mel spectrogram in dB scale (2D numpy array)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0  # Power spectrogram
    )

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Normalize spectrogram to zero mean and unit variance.

    Args:
        spec: Mel spectrogram (2D numpy array)

    Returns:
        Normalized spectrogram
    """
    mean = np.mean(spec)
    std = np.std(spec)

    # Avoid division by zero
    if std < 1e-10:
        std = 1.0

    normalized = (spec - mean) / std
    return normalized


def preprocess_and_save(audio_path: str,
                        output_path: str,
                        verbose: bool = False) -> bool:
    """
    Full preprocessing pipeline: load → mel spectrogram → normalize → save

    Args:
        audio_path: Path to input audio file
        output_path: Path to save preprocessed .npy file
        verbose: Print progress information

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load audio
        if verbose:
            print(f"Loading: {Path(audio_path).name}")
        audio = load_audio(audio_path)

        # Convert to mel spectrogram
        if verbose:
            print(f"  Converting to mel spectrogram...")
        mel_spec = audio_to_mel_spectrogram(audio)

        # Normalize
        if verbose:
            print(f"  Normalizing...")
        mel_spec_norm = normalize_spectrogram(mel_spec)

        # Save as .npy
        if verbose:
            print(f"  Saving to: {Path(output_path).name}")
            print(f"  Shape: {mel_spec_norm.shape}")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, mel_spec_norm)

        return True

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False


def get_spectrogram_shape() -> Tuple[int, int]:
    """
    Calculate expected mel spectrogram shape.

    Note: The actual shape from librosa is (128, 938) for our configuration.

    Returns:
        Tuple of (n_mels, n_frames)
    """
    # Empirically determined from librosa
    # With SR=16000, duration=30s, n_fft=2048, hop=512
    # 30s * 16000 Hz = 480,000 samples
    # frames = ceil(samples / hop_length) = ceil(480000 / 512) = 938
    return (N_MELS, 938)


def verify_preprocessing(npy_path: str) -> None:
    """
    Load and verify a preprocessed .npy file.

    Args:
        npy_path: Path to .npy file
    """
    spec = np.load(npy_path)
    expected_shape = get_spectrogram_shape()

    print(f"\nVerifying: {Path(npy_path).name}")
    print(f"  Shape: {spec.shape}")
    print(f"  Expected: {expected_shape}")
    print(f"  Mean: {np.mean(spec):.4f}")
    print(f"  Std: {np.std(spec):.4f}")
    print(f"  Min: {np.min(spec):.4f}")
    print(f"  Max: {np.max(spec):.4f}")

    if spec.shape == expected_shape:
        print(f"  ✅ Shape matches!")
    else:
        print(f"  ❌ Shape mismatch!")


if __name__ == "__main__":
    # Test preprocessing on a single file
    import glob

    print("Testing Audio Preprocessing Module")
    print("=" * 70)

    # Test with a GTZAN file (real song)
    gtzan_files = glob.glob(
        "/Users/stanleycliu/.cache/kagglehub/datasets/andradaolteanu/"
        "gtzan-dataset-music-genre-classification/versions/1/Data/"
        "genres_original/blues/*.wav"
    )

    if gtzan_files:
        test_file = gtzan_files[0]
        output_file = "test_spectrogram.npy"

        print(f"\nTest file: {Path(test_file).name}")
        print(f"Processing...")

        success = preprocess_and_save(test_file, output_file, verbose=True)

        if success:
            print("\n✅ Preprocessing successful!")
            verify_preprocessing(output_file)

            # Clean up test file
            Path(output_file).unlink()
            print(f"\nTest file removed.")
        else:
            print("\n❌ Preprocessing failed!")
    else:
        print("No GTZAN files found for testing.")

    print(f"\nExpected spectrogram shape: {get_spectrogram_shape()}")
