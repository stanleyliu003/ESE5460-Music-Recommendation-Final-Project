# ESE5460-Music-Recommendation-Final-Project

## Team 50

- Matt Park  
- Spencer Ware  
- Hassan Rizwan  
- Stanley Liu  

## Project Title

**Classifying AI-Generated vs Non-AI-Generated Music with Top-K Human Song Recommendations**

## Abstract

This project builds a **binary audio classifier** that predicts whether a given music track is **AI-generated** or **human-composed**.  
For any songs identified as AI-generated above a chosen confidence threshold, the system will run an **embedding-based recommendation** pipeline to return **top-\(K\)** semantically similar **human-composed** songs.  

The end goal is twofold:

1. **Authenticity** – Help listeners and platforms identify AI-generated content.  
2. **Discoverability** – Use an AI-generated song that a listener enjoys as a *sonic reference* to recommend human artists producing similar music.  

This promotes transparency around AI content while simultaneously boosting exposure for human musicians in a personalized, taste-aware way.

## Background and Motivation

Recent advances in generative AI for music (e.g., Suno, Udio) have led to an **explosion of AI-generated tracks** on streaming platforms and social media.  
User studies and platform data suggest that **most listeners (up to ~97%) cannot reliably distinguish** AI-generated music from human-composed tracks.  

This creates two major challenges:

- **Authenticity**:  
  Listeners who wish to support human artists have **no reliable signal** about whether a track is AI-generated or not.

- **Discoverability**:  
  A listener might genuinely enjoy the style, mood, or production of an AI-generated track, but currently has **no direct pathway** to discover *human* artists creating similar music.

This project addresses both problems:

- A **binary classifier** provides a tool for **flagging AI-generated content**.  
- A **recommendation system**, conditioned on the AI song’s audio representation, **maps listeners back to similar human-made songs**, improving discoverability and support for human artists.

## Dataset

We use two primary datasets:

### 1. SONICS Dataset
- ~49,000+ AI-generated songs
- ~48,000+ human-generated songs
- Kaggle: `https://www.kaggle.com/datasets/awsaf49/sonics-dataset`
- Hugging Face: `https://huggingface.co/datasets/awsaf49/sonics`

### 2. GTZAN Dataset
- Music genre classification dataset
- 10 genres, 100 tracks each (1000 total)
- Kaggle: `https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification`

### Setup Instructions

**Important**: The datasets are NOT included in this repository due to their large size. Follow these steps to download them:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download datasets:
   ```bash
   python download_dataset.py
   ```

This script will download both required datasets to your local machine via `kagglehub`. The datasets are cached locally and don't need to be re-downloaded.

## High-Level Approach

The system consists of two main components:

1. **Model 1 – Binary Audio Classifier**  
   - Input: Mel spectrogram representation of an audio clip.  
   - Output: Probability that the track is **AI-generated** vs **human-generated**.

2. **Model 2 – Top-\(K\) Human Song Recommender**  
   - Triggered when Model 1 predicts that a track is **AI-generated** above a specified threshold.  
   - Uses **OpenL3** audio embeddings and a **K-Nearest Neighbors (KNN)** search to retrieve semantically similar **human** tracks.

While these two models can have separate inputs and architectures, they will share a common representation space based on spectrograms and/or embeddings, enabling smooth handoff from detection to recommendation.

## Technical Design

### Data Preprocessing

- Download the SONICS dataset (AI and human tracks).  
- For each audio file:
  - Extract the **first 30 seconds** of audio.  
  - Convert the 1D waveform into a **2D Mel spectrogram**.  
  - Convert Mel spectrogram amplitudes to **decibels (dB)** to enhance contrast and dynamic range.  

These spectrograms serve as inputs to:

- The **transformer-based binary classifier** (Model 1).  
- The **OpenL3 embedding model** (Model 2).

### Model 1: Binary Audio Classifier

**Goal**: Predict whether a given track is AI-generated or human-generated.

**Input**: Mel spectrogram (e.g., 30-second clip).  

**Planned architecture** (subject to iteration):

- **Tokenization** of the spectrogram along temporal and/or spectral slices (e.g., patches or frames).  
- **Positional encoding** to preserve temporal order and/or frequency structure.  
- **Transformer encoder** layers to model long-range dependencies in time–frequency space.  
- **Pooling** (e.g., average pooling over token dimension) to obtain a fixed-length representation.  
- **Final classification head** (e.g., one or two linear layers) outputting \(P(\text{AI})\) and \(P(\text{human})\).

Predictions above a confidence threshold on the **AI** class will be passed to the recommendation pipeline.

### Model 2: Top-\(K\) Human Song Recommendation

**Goal**: Given an audio track predicted to be AI-generated, recommend **top-\(K\)** similar **human-composed** tracks.

**Embedding model**:  
- Use **OpenL3**, a pre-trained audio embedding model trained on millions of audio files.  
- Input: audio spectrogram (or audio waveform, depending on configuration).  
- Output: a **512-dimensional vector** representation (a “sonic fingerprint”) capturing timbral, rhythmic, and harmonic characteristics.

**Offline processing**:

- Run the **~48,000 human-generated tracks** from SONICS through OpenL3.  
- Store the resulting 512-D vectors in a **vector database** (e.g., a NumPy array or more scalable vector index).  

**Online / query-time processing**:

- When Model 1 flags a track as AI-generated:
  - Compute its OpenL3 embedding (512-D vector).  
  - Use **K-Nearest Neighbors (KNN)** with **cosine similarity** to search the human-song embedding database.  
  - Return the **top-\(K\)** nearest human tracks (e.g., \(K = 5\)) as recommendations.

## Four-Week Timeline

### Week 1: Data Acquisition and Preprocessing

- Download the SONICS dataset (AI + human tracks).  
- Implement audio loading and trimming to the first 30 seconds.  
- Compute **Mel spectrograms** and convert amplitudes to **dB**.  
- Organize processed data into train/validation/test splits for the classifier.  

### Week 2: Model 1 – Binary Audio Classifier

- Design and implement the **transformer-based classifier**:
  - Define spectrogram tokenization scheme (temporal / spectral patches).  
  - Add positional encodings.  
  - Implement transformer encoder layers and pooling.  
  - Train and tune hyperparameters on SONICS.  
- Evaluate performance using accuracy, precision/recall, F1-score, and ROC-AUC.  

### Week 3: Model 2 – Top-\(K\) Recommendation System

- Integrate **OpenL3** for embedding extraction.  
- Compute and store **512-D embeddings** for all human-generated tracks.  
- Implement **KNN search** over the embedding database with **cosine similarity**.  
- Define the end-to-end inference pipeline:
  - If a track is predicted AI-generated above threshold → run recommender → output **top-\(K\)** human tracks.  

### Week 4: Refinement, Debugging, and Write-Up

- Debug and improve both classifier and recommender for better accuracy and robustness.  
- Perform ablation studies (e.g., different spectrogram parameters, embedding configurations, or thresholds).  
- Conduct qualitative listening tests and case studies.  
- Write the **final project paper/report**, including methodology, experiments, and analysis.  

## Open Questions

- **Model coupling**:  
  Currently, the classifier and recommender have separate inputs and architectures.  
  - Should we introduce a shared embedding space or shared backbone to more tightly couple the **classification** and **recommendation** stages?  
  - Would jointly training or fine-tuning with a multi-task objective (classification + similarity) improve performance?

- **Additional embeddings / features**:  
  - Are there alternative or complementary embedding models (e.g., CLAP, other music-specific encoders) that might capture genre, mood, or vocal presence more robustly?  
  - Should we explore additional transformations such as chroma features, tempo/beat features, or higher-resolution time–frequency representations?  
  - Would ensembling multiple embeddings improve recommendation quality without prohibitive compute cost?

These questions will guide experimentation beyond the core baseline system.

## Resources

- SONICS dataset on Kaggle: `https://www.kaggle.com/datasets/awsaf49/sonics-dataset`  
- SONICS dataset on Hugging Face: `https://huggingface.co/datasets/awsaf49/sonics`  
- Article on AI music and streaming platforms (e.g., The Guardian):  
  `https://www.theguardian.com/technology/2025/nov/13/ai-music-spotify-billboard-charts`  
- OpenL3 GitHub repository: `https://github.com/marl/openl3`  


