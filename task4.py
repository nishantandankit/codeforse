import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Create a Sample Dataset ---
# A user-item matrix where rows are users, columns are movies, and values are ratings (1-5).
# np.nan means the user has not rated the movie.
data = {
    'Movie A': [5, 4, 1, np.nan],
    'Movie B': [4, 5, 2, 1],
    'Movie C': [np.nan, 3, 5, 4],
    'Movie D': [1, 1, 5, 5],
    'Movie E': [2, 2, np.nan, 4]
}
df = pd.DataFrame(data, index=['User 1', 'User 2', 'User 3', 'User 4'])

print("--- User-Movie Ratings Matrix ---")
print(df)

# --- 2. Preprocessing and Similarity Calculation ---
# For calculating similarity, we must handle missing values.
# Filling NaNs with 0 assumes a user has no interest in an unrated movie.
df_filled = df.fillna(0)

# Calculate the cosine similarity between all users based on their rating vectors.
user_similarity = cosine_similarity(df_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=df.index, columns=df.index)

print("\n--- User Similarity Matrix ---")
print(user_similarity_df)

# --- 3. Generate Recommendations ---
def get_recommendations(user_id, num_recommendations=2):
    """Generates movie recommendations for a specific user."""
    print(f"\n--- Generating recommendations for {user_id} ---")

    # 1. Find users most similar to our target user.
    # We sort them and drop the first one (which is the user themselves).
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:]

    # 2. Find the movies the target user has NOT yet rated.
    movies_not_watched = df.loc[user_id][df.loc[user_id].isnull()].index

    if len(movies_not_watched) == 0:
        return "This user has rated all available movies."

    # 3. For each movie the user hasn't seen, calculate a weighted score
    # based on ratings from similar users.
    recommendations = {}
    for movie in movies_not_watched:
        weighted_sum = 0
        similarity_sum = 0
        for other_user, similarity_score in similar_users.items():
            # Check if the similar user has rated this movie
            if pd.notna(df.loc[other_user, movie]):
                weighted_sum += similarity_score * df.loc[other_user, movie]
                similarity_sum += similarity_score

        # To avoid division by zero
        if similarity_sum > 0:
            recommendations[movie] = weighted_sum / similarity_sum

    # 4. Sort the recommended movies by their calculated score.
    sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)

    return sorted_recommendations[:num_recommendations]

# --- 4. Showcasing Recommendation Results ---
# Get recommendations for a user who hasn't seen Movie C and Movie E
user1_recs = get_recommendations('User 1')
print(f"Top recommendations for User 1: {user1_recs}")

# Get recommendations for a user who hasn't seen Movie A and Movie E
user3_recs = get_recommendations('User 3')
print(f"Top recommendations for User 3: {user3_recs}")
