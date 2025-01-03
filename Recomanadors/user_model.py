import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from svd import arrodonir
from extreure_metadata import movie_finder, metadata_extractor

# Similarity computation
def compute_user_similarity(user_movie_matrix):
    filled_matrix = user_movie_matrix.fillna(0)  # Replace NaN with 0
    similarity_matrix = cosine_similarity(filled_matrix)
    return pd.DataFrame(similarity_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Variance weights
def compute_variance_weights(user_movie_matrix):
    return user_movie_matrix.var(axis=0).apply(np.log1p)

# Predict single movie rating
def predict_single_movie(user_id, movie_id, user_movie_matrix, user_similarity_matrix, jaccard_similarity_matrix, variance_weights, regularization=1e-3, similarity_threshold=0.75):
    user_ratings = user_movie_matrix.loc[user_id]
    valid_users = user_movie_matrix.index[user_movie_matrix[movie_id].notna() & (user_movie_matrix.index != user_id)]

    if len(valid_users) == 0:
        return np.nan  # No valid users for prediction

    # Filter users by similarity threshold
    similarities = user_similarity_matrix.loc[user_id, valid_users]
    valid_users = valid_users[similarities >= similarity_threshold]

    if len(valid_users) == 0:
        return np.nan  # No valid users for prediction

    # similarities = user_similarity_matrix.loc[user_id, valid_users]
    # jaccard = jaccard_similarity_matrix.loc[user_id, valid_users]
    # combined = 0.5 * similarities + 0.5 * jaccard
    weights = (similarities * variance_weights[movie_id]) / (regularization + np.abs(similarities))  # Regularization

    ratings = user_movie_matrix.loc[valid_users, movie_id]
    weighted_sum = (weights * ratings).sum()
    normalization_factor = weights.sum() + 1e-8  # Avoid division by zero

    predicted_rating = weighted_sum / normalization_factor

    # Normalize predictions to range [0, 5]
    return predicted_rating

# Evaluate model
def evaluate_model(user_movie_matrix, user_similarity_matrix, jaccard_similarity_matrix,variance_weights):
    actual_ratings = []
    predicted_ratings = []

    for user_id in user_movie_matrix.index:
        for movie_id in user_movie_matrix.columns:
            actual_rating = user_movie_matrix.loc[user_id, movie_id]
            if not np.isnan(actual_rating):
                predicted_rating = predict_single_movie(
                    user_id, movie_id, user_movie_matrix, user_similarity_matrix, jaccard_similarity_matrix, variance_weights
                )
                if not np.isnan(predicted_rating):
                    actual_ratings.append(actual_rating)
                    predicted_ratings.append(predicted_rating)

    # Normalitzar les dades
    predicted_ratings = np.array(predicted_ratings)
    min_rating = predicted_ratings.min()
    max_rating = predicted_ratings.max()
    predicted_ratings = 0 + 5 * (predicted_ratings - min_rating) / (max_rating - min_rating)
    predicted_ratings = [arrodonir(valor) for valor in predicted_ratings] 

    # Calculate evaluation metrics
    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    mse = mean_squared_error(actual_ratings, predicted_ratings)
    rmse = np.sqrt(mse)

    print(f"Model Evaluation:\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.hist([actual_ratings, predicted_ratings], label=['Actual', 'Predicted'], bins=20, alpha=0.7)
    plt.legend()
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Actual vs Predicted Ratings')
    plt.show()

# Recommend movies for a user
def recommend_movies(user_id, user_movie_matrix, user_similarity_matrix, jaccard_similarity_matrix,variance_weights):
    user_ratings = user_movie_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings.isna()].index

    predictions = {}
    for movie_id in unrated_movies:
        predictions[movie_id] = predict_single_movie(
            user_id, movie_id, user_movie_matrix, user_similarity_matrix, jaccard_similarity_matrix, variance_weights
        )

    ratings = list(predictions.values())
    min_rating = min(ratings)
    max_rating = max(ratings)
    for movie in predictions:
        predictions[movie] = 0 + 5 * (predictions[movie] - min_rating) / (max_rating - min_rating)

    # Sort movies by predicted ratings
    recommended_movies = pd.Series(predictions).sort_values(ascending=False)
    tmp = sorted(list(predictions.items()), key= lambda x: x[1], reverse=True)
    recommended_movies = [tupla[0] for tupla in tmp]
    return recommended_movies

def main_user():
    # Example dataset (replace with actual data for larger cases)
    # Inicialització dataframes
    user = int(input("Introdueix ID d'usuari per recomanar-li pel·lícules: "))
    ratings = pd.read_csv('./Data/ratings_small.csv') # movieid = int64
    movies = pd.read_csv('./Data/movies_metadata.csv', low_memory=False)

    user_movie_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    )

    similarity_matrix_path = './Data/user_similarity_matrix.csv'
    jaccard_matrix_path = './Data/jaccard_similarity_matrix.csv'

    user_similarity_matrix = pd.read_csv(similarity_matrix_path, index_col=0)
    user_similarity_matrix.index = user_similarity_matrix.index.astype(float)
    user_similarity_matrix.columns = user_similarity_matrix.columns.astype(float)
    jaccard_similarity_matrix = pd.DataFrame()
    # jaccard_similarity_matrix = pd.read_csv(jaccard_matrix_path, index_col=0)
    # jaccard_similarity_matrix.index = jaccard_similarity_matrix.index.astype(float)
    # jaccard_similarity_matrix.columns = jaccard_similarity_matrix.columns.astype(float)
    variance_weights = compute_variance_weights(user_movie_matrix)

    evaluate_model(user_movie_matrix, user_similarity_matrix, jaccard_similarity_matrix, variance_weights)

    # user = 53

    # Get recommendations for a user
    recommendations = recommend_movies(user, user_movie_matrix, user_similarity_matrix, jaccard_similarity_matrix, variance_weights)
    ids = movie_finder(recommendations, movies, 10)
    metadata_extractor(ids, movies)
    # print(f"Recommendations for {user}:")
    # print(recommendations)

# Example usage
if __name__ == "__main__":
    main_user()