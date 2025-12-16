# Data Processing Scripts

This directory contains scripts for data exploration, preprocessing, and dataset management.

## Scripts

### 1. `explore_datasets.py` ✅ COMPLETED

Explores both SONICS and GTZAN datasets to understand structure and properties.

**Usage**:
```bash
python src/data/explore_datasets.py
```

**What it does**:
- Counts total files in each dataset
- Shows sample filenames
- Checks audio properties (sample rate, duration, channels)
- Generates summary statistics

**Output**:
- Console output with dataset statistics
- Identifies SONICS: 21,230 AI songs (.mp3, 16kHz)
- Identifies GTZAN: 1,000 real songs (.wav, 22.05kHz)

---

### 2. `audio_preprocessing.py` (TODO: Phase 2)

Handles audio loading and mel spectrogram conversion.

**Functions to implement**:
- `load_audio()` - Load and resample audio to 22,050 Hz
- `audio_to_mel_spectrogram()` - Convert audio → mel spectrogram
- `normalize_spectrogram()` - Normalize to mean=0, std=1
- `preprocess_and_save()` - Full pipeline + save as .npy

---

### 3. `create_splits.py` (TODO: Phase 2)

Creates train/validation/test splits.

**What it will do**:
- Randomly sample 1,000 AI songs from SONICS
- Use all 1,000 real songs from GTZAN
- Create stratified 70/15/15 split
- Save indices to `data/splits/split_indices.json`

---

### 4. `dataset.py` (TODO: Phase 3)

PyTorch Dataset class for loading preprocessed spectrograms.

**What it will do**:
- Load preprocessed `.npy` files
- Return (spectrogram, label) pairs
- Support data augmentation (optional)

---

## Dataset Paths

**SONICS (AI songs)**:
```
/Users/stanleycliu/.cache/kagglehub/datasets/awsaf49/sonics-dataset/versions/2/fake_songs
```

**GTZAN (Real songs)**:
```
/Users/stanleycliu/.cache/kagglehub/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/genres_original
```

---

## Next Steps

Run Phase 2 preprocessing:
1. Implement `audio_preprocessing.py`
2. Implement `create_splits.py`
3. Run preprocessing on 2,000 songs (1K AI + 1K real)
4. Save spectrograms to `data/processed/`
