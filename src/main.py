from data import load_and_explore_data, data_preprocessing, combined_feature
from vectorizer import create_tfidf_tensor
from similarity import compute_recommendation
from recommender import save_recommendation, recommend_songs
from pathlib import Path
import argparse

# Interactive app
def interactive_cli():
    while True:
        user_input = input("\nEnter a song title (or 'exit' to quit): ").strip()
    
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
    
        recs = recommend_songs(user_input)
    
        if recs is None:
            print(f"We don't have recommendations for {user_input}")
        else:
            print(f"\nTop 10 recommendations for '{user_input}': ")
            for i, song in enumerate(recs, start = 1):
                print(f"{i}. {song}")

def main():
    # Create parser
    parser= argparse.ArgumentParser(
        description = "GPU-Accelerated Song Recommendation System"
    )

    parser.add_argument(
        "--build",
        action = "store_true",
        help = "Rebuild song recommendations"
    )
    
    args = parser.parse_args()

    csv_path = Path("outputs/song_recommendations.csv")

    # Rebuild recommendations if requested or CSV is missing
    if args.build or not csv_path.exists():
        selected_songs = load_and_explore_data()
        selected_songs = data_preprocessing(selected_songs)
        selected_songs = combined_feature(selected_songs)

        X = create_tfidf_tensor(selected_songs)

        recommendations = compute_recommendation(X, selected_songs)

        save_recommendation(recommendations)
    
    interactive_cli()

if __name__ == "__main__":
    main()