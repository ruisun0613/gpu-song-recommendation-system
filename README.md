# GPU-Accelerated Song Recommendation System

A content-based music recommendation system that uses TF-IDF text features and PyTorch GPU acceleration to generate Top-10 recommendations from Amazon Digital Music metadata.

## Overview

This project builds a content-based recommendation system using metadata such as:

- Title
- Description
- Category
- Brand
- Feature

The metadata is cleaned and combined into a single text representation. TF-IDF is then used to convert the text into numerical feature vectors.

Similarity between music items is computed using PyTorch matrix multiplication. When CUDA is available, the feature matrix is moved to the GPU and FP16 precision is used to reduce VRAM usage.

To avoid storing a full pairwise similarity matrix, similarity computation is performed in batches and only the Top-10 most similar items are retained.

## Tech Stack

- Python
- PyTorch
- Pandas
- Scikit-learn
- TF-IDF
- CUDA

## Project Structure

```text
gpu-song-recommender/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── vectorizer.py
│   ├── similarity.py
│   ├── recommender.py
│   └── main.py
│
├── dataset/
├── outputs/
├── .gitignore
├── README.md
└── requirements.txt
```

## Pipeline

```text
Amazon Digital Music Metadata
        ↓
Data Cleaning & Preprocessing
        ↓
Combined Text Features
        ↓
TF-IDF Vectorization
        ↓
PyTorch Tensor
        ↓
GPU / CPU Batch Similarity
        ↓
Top-10 Recommendations
```

## Recommendation Approach

Cosine similarity is calculated between normalized TF-IDF vectors.

Instead of creating and storing the complete item-to-item similarity matrix, the system processes items in batches:

1. Select a batch of TF-IDF vectors.
2. Compute similarity against the full feature matrix.
3. Remove self-matches.
4. Select the Top-10 most similar items.
5. Store only the recommendation results.

This approach reduces the memory required for the similarity results while still allowing recommendations to be generated for the full dataset.

## Installation

```bash
python -m venv .venv
```

Activate the virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

- pandas
- torch
- scikit-learn

## Status

This project is currently being refactored from the original implementation into a modular structure for improved readability and maintainability.