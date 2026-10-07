from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import torch
import torch.nn.functional as F

# TF-IDF vectorization using sklearn
def create_tfidf_tensor(selected_songs: pd.DataFrame) -> torch.Tensor:
    tfidf = TfidfVectorizer(stop_words = "english", max_features = 5000)
    tfidf_matrix = tfidf.fit_transform(selected_songs["combined_features"])

    # result
    print("TF-IDF matrix shape:", tfidf_matrix.shape)

    # Check cuda or cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Convert TF-IDF matrix to PyTorch tensor
    X = torch.tensor(tfidf_matrix.toarray(), dtype = torch.float32)

    if device == "cuda":
        X = X.to(device)
        X = F.normalize(X, p = 2, dim = 1)
        X = X.half      # reduce VRAM usage
    else:
        X = X.to(device)
        X = F.normalize(X, p = 2, dim = 1)

    return X