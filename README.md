# GPU-Accelerated Song Recommendation System

A content-based music recommendation system built with Python, Scikit-learn, and PyTorch.

The system uses TF-IDF to extract textual features from Amazon Digital Music metadata and computes item-to-item cosine similarity using GPU-accelerated matrix multiplication.

It generates Top-10 recommendations for 59,572 music items and supports interactive searches through a command-line interface.

## Features

- **Content-Based Recommendations:** Generates recommendations based on music metadata, including titles, descriptions, categories, brands, and features.
- **GPU Acceleration:** Uses PyTorch and CUDA to accelerate similarity calculations.
- **FP16 Optimization:** Uses half-precision tensors on CUDA devices to reduce GPU memory consumption.
- **Batch Processing:** Computes similarity scores in batches without storing the complete pairwise similarity matrix.
- **Top-10 Recommendations:** Identifies the 10 most similar items for each music entry.
- **Precomputed Results:** Saves recommendations to CSV for fast subsequent searches.
- **Interactive CLI:** Allows users to search for recommendations by music title.
- **CPU Fallback:** Supports recommendation generation on systems without CUDA-compatible GPUs.
- **Modular Architecture:** Separates data processing, feature extraction, similarity computation, and recommendation functionality.

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Pandas | Data processing and CSV operations |
| Scikit-learn | TF-IDF feature extraction |
| PyTorch | Tensor operations and similarity computation |
| CUDA | NVIDIA GPU acceleration |
| FP16 | Reduced GPU memory usage |
| argparse | Command-line argument handling |

## Project Structure

```text
gpu-song-recommendation-system/
│
├── src/
│   ├── __init__.py
│   ├── data.py              # Data loading and preprocessing
│   ├── vectorizer.py        # TF-IDF vectorization and tensor conversion
│   ├── similarity.py        # Batch similarity computation
│   ├── recommender.py       # Recommendation storage and retrieval
│   └── main.py              # Application entry point
│
├── dataset/
│   └── meta_Digital_Music.json   # Original dataset (not included)
│
├── outputs/
│   └── song_recommendations.csv  # Precomputed recommendations
│
├── .gitignore
├── README.md
└── requirements.txt
```

## How It Works

The recommendation pipeline consists of six stages:

```text
Amazon Digital Music Metadata
             |
             v
    Data Preprocessing
             |
             v
    Feature Combination
             |
             v
    TF-IDF Vectorization
             |
             v
    PyTorch Tensor Conversion
             |
             v
    GPU / CPU Batch Similarity
             |
             v
    Top-10 Recommendations
             |
             v
       CSV Storage
             |
             v
     Interactive Search
```

### 1. Data Preprocessing

The system loads Amazon Digital Music metadata and extracts the following fields:

- Title
- Description
- Category
- Brand
- Feature

Missing values, empty titles, and duplicate titles are handled during preprocessing.

Text features are combined into a single representation and normalized through lowercase conversion and basic text cleaning.

### 2. TF-IDF Vectorization

The combined text is converted into numerical feature vectors using Scikit-learn's `TfidfVectorizer`.

The vectorizer uses:

- English stop-word removal
- A maximum of 5,000 features
- TF-IDF weighting

The resulting feature matrix contains:

```text
59,572 items × 5,000 features
```

### 3. GPU Acceleration

The TF-IDF matrix is converted into a PyTorch tensor.

When CUDA is available, the tensor is transferred to the GPU and converted to FP16 precision to reduce memory consumption.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

If CUDA is unavailable, the system automatically uses the CPU.

### 4. Batch Similarity Computation

Cosine similarity is calculated between normalized TF-IDF vectors.

Instead of storing a complete similarity matrix containing approximately 3.55 billion item pairs, the system processes the data in batches.

For each batch:

1. Compute similarity scores against the full feature matrix.
2. Exclude self-matches.
3. Select the Top-10 most similar items.
4. Store the recommendation results.
5. Continue processing the next batch.

The default batch size is 256.

This approach reduces the memory required for intermediate similarity results.

### 5. Recommendation Storage

Generated recommendations are saved to:

```text
outputs/song_recommendations.csv
```

Each row contains an item title and its Top-10 recommended titles.

The application can reuse this file without repeating the entire recommendation pipeline.

### 6. Interactive Search

Users can enter a music title through the command-line interface.

The application performs a case-insensitive exact-title lookup and displays up to 10 recommendations.

## Dataset

This project uses Amazon Digital Music product metadata.

The dataset format is consistent with the Amazon Review Data (2018) collection maintained by the UCSD research team.

**Dataset source:**

[Amazon Review Data — UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)

### Dataset Statistics

| Description | Count |
|-------------|-------|
| Original records | 74,347 |
| Records after preprocessing | 59,572 |
| TF-IDF features | 5,000 |
| Recommendations per item | 10 |

The original dataset is not included in this repository.

To rebuild recommendations, obtain the Digital Music metadata dataset, extract it if necessary, and place the JSON file at:

```text
dataset/meta_Digital_Music.json
```

The application expects a JSON Lines file containing one JSON object per line.

**Note:** The dataset contains Amazon music product metadata, which may include albums and other music-related products rather than individual songs exclusively.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/gpu-song-recommendation-system.git

cd gpu-song-recommendation-system
```

Replace `YOUR_USERNAME` with the appropriate GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment.

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install PyTorch

For GPU-accelerated recommendation generation on supported NVIDIA GPUs, install a CUDA-enabled version of PyTorch.

The project was tested with:

- NVIDIA GeForce RTX 5080
- PyTorch 2.14.1
- CUDA 13.0
- GPU compute capability `sm_120`

Installation command for the tested configuration:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
```

Other hardware configurations may require a different PyTorch build.

For installation instructions, refer to the official PyTorch website.

## Usage

### 1. Search Existing Recommendations

If the precomputed recommendation CSV is available, start the application with:

```bash
python src/main.py
```

The program will enter interactive search mode:

```text
Enter a song title (or 'exit' to quit):
```

Enter a title to display its Top-10 recommendations.

Type `exit` to close the application.

### 2. Rebuild Recommendations

To regenerate recommendations from the original dataset:

```bash
python src/main.py --build
```

This command will:

1. Load and preprocess the dataset.
2. Generate TF-IDF features.
3. Transfer tensors to the GPU when CUDA is available.
4. Compute Top-10 recommendations in batches.
5. Save the results to CSV.
6. Start the interactive search interface.

### 3. Automatic Dataset Processing

If the recommendation CSV does not exist, the application automatically runs the recommendation pipeline before starting interactive mode.

Rebuilding requires the original dataset to be available in the `dataset/` directory.

### 4. Command-Line Help

```bash
python src/main.py --help
```

Available options:

```text
-h, --help    Show the help message and exit
--build       Rebuild song recommendations
```

## Performance and Memory Optimization

The system implements several optimizations:

**GPU Matrix Multiplication**

PyTorch performs batched matrix multiplication on CUDA-compatible GPUs.

**FP16 Precision**

CUDA tensors use half-precision floating-point values to reduce GPU memory consumption.

**Batch Processing**

Similarity scores are calculated for a subset of items at a time, avoiding storage of the complete pairwise similarity matrix.

**Top-K Selection**

Only the 10 highest-scoring recommendations are retained for each item.

**Precomputed Recommendations**

Previously generated results can be loaded directly from CSV, avoiding unnecessary recomputation.

## Limitations

- Recommendations are based on textual metadata rather than audio characteristics.
- The system does not use listening history, user ratings, or collaborative filtering.
- TF-IDF primarily captures keyword-based similarity rather than deeper semantic relationships.
- Similar product descriptions may produce recommendations that are not musically similar.
- Generating recommendations for the full dataset requires substantial computation.
- The current implementation converts the sparse TF-IDF matrix into a dense tensor, which increases memory usage.
- The interactive search requires an exact title match, although matching is case-insensitive.

## Future Improvements

Potential improvements include:

- Semantic embeddings for more meaningful recommendations.
- Approximate nearest-neighbor search for improved scalability.
- A web interface for interactive recommendations.
- Support for partial-title searches.
- Additional evaluation metrics for recommendation quality.

## Project Status

**Completed**

The system has been refactored into a modular Python application.

The recommendation pipeline, GPU acceleration, CSV storage, command-line interface, and automatic cache detection have been implemented and tested.

The full recommendation pipeline was successfully executed using an NVIDIA GeForce RTX 5080.