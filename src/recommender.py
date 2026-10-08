import pandas as pd

# Save recommendation
def save_recommendation(recommendations):
    recommendation_df = pd.DataFrame(recommendations)
    recommendation_df.to_csv("outputs/song_recommendations.csv", index = False, encoding = 'utf-8')

    print("\nRecommendations saved to song_recommendations.csv")

# Recommender function
def recommend_songs(song_title):
    # Load saved recommendations for the app
    app_recommendations = pd.read_csv("outputs/song_recommendations.csv")

    match = app_recommendations["title"].str.lower() == song_title.lower()
    if not match.any():
        return None

    row = app_recommendations[match].iloc[0]
    results = []

    for i in range(1, 11):
        col_name = f"rec_{i}"
        if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip():
            results.append(row[col_name])

    return results