import json
import re
import pandas as pd

# Load json lines file into dataframe
def load_and_explore_data() -> pd.DataFrame:
    data = []
    with open("dataset/meta_Digital_Music.json", "r", encoding = "utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    songs = pd.DataFrame(data)

    # Explore data
    print("Data Exploration")
    print(f"Data Shape: {songs.shape}")
    print(f"\nColumns: {songs.columns.tolist()}")
    print(f"\nNull values per column: {songs.isnull().sum()}")

    print("\nEmpty string count per column:")
    for col in songs.columns:
        empty_count = songs[col].astype(str).str.strip().eq("").sum()
        print(f"{col}: {empty_count}")

    # Select useful columns
    selected_columns = [col for col in ["title", "description", "category", "brand", "feature"] if col in songs.columns]
    selected_songs = songs[selected_columns].copy()

    return selected_songs
# Helper function to convert lists to text
def list_to_text(x) -> str:
    if isinstance(x, list):
        output = []
        for item in x:
            if isinstance(item, list):
                output.extend([str(v) for v in item])
            else:
                output.append(str(item))
        return " ".join(output)
    return str(x)

# Data preprocessing
def data_preprocessing(selected_songs: pd.DataFrame) -> pd.DataFrame:
    # Clean title column
    selected_songs["title"] = selected_songs["title"].fillna("").astype(str).str.strip()
    selected_songs = selected_songs[selected_songs["title"] != ""]
    selected_songs = selected_songs.drop_duplicates(subset = ["title"]).reset_index(drop = True)

    # Prepare text columns
    for col in ["description", "category", "brand", "feature"]:
        if col not in selected_songs.columns:
            selected_songs[col] = ""
        selected_songs[col] = selected_songs[col].fillna("").apply(list_to_text)

    return selected_songs

# Text cleaning function
def clean_text(text) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ",text).strip()
    return text

# Create combined feature space
def combined_feature(selected_songs) -> pd.DataFrame:
    selected_songs["combined_features"] = (
        selected_songs["title"] + " " + 
        selected_songs["description"] + " " +
        selected_songs["category"] + " " +
        selected_songs["brand"] + " " +
        selected_songs["feature"]
    )

    selected_songs["combined_features"] = selected_songs["combined_features"].apply(clean_text)
    selected_songs = selected_songs[selected_songs["combined_features"] != ""].reset_index(drop = True)

    print(f"\nClean dataset shape: {selected_songs.shape}")

    return selected_songs