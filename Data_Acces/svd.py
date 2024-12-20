import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix

from extreure_metadata import movie_finder, metadata_extractor
from user_to_user import print_recommendations


# Cargar los datos
ratings = pd.read_csv('./Data/ratings.csv')
ratings = ratings[ratings['userId'] <= 1000]  # Solo usuarios con ID <= 1000
ratings = ratings[ratings['movieId'] <= 1000]  # Solo películas con ID <= 1000

movies = pd.read_csv('./Data/movies_metadata.csv')

# Dividir en entrenamiento (80%) y prueba (20%)
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

user_movie_matrix_sparse = csr_matrix(
    (train_data['rating'], (train_data['userId'], train_data['movieId']))
)

rmse_values = []
# k_values = [1, 2, 5, 10, 15, 20, 30, 50, 100]
k_values = [10]

for k in k_values:
    U, sigma, Vt = svds(user_movie_matrix_sparse, k=k)
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    # Convertir la matriz dispersa a DataFrame denso
    predicted_ratings_df = pd.DataFrame(
        predicted_ratings, 
        index=np.arange(user_movie_matrix_sparse.shape[0]),  # Asignar índices de usuarios
        columns=np.arange(user_movie_matrix_sparse.shape[1])  # Asignar índices de películas
    )

    predictions = []
    true_ratings = []
    for user, movie, actual in zip(test_data['userId'], test_data['movieId'], test_data['rating']):
        if user in predicted_ratings_df.index and movie in predicted_ratings_df.columns:
            pred = predicted_ratings_df.loc[user, movie]
        else:
            pred = 0
        predictions.append(pred)
        true_ratings.append(actual)
    
    rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
    rmse_values.append(rmse)
    print(f"k = {k}, RMSE = {rmse:.4f}")

def recommend_movies(user_id, predicted_ratings_df, original_ratings, n=5):
    # Películas ya vistas por el usuario
    seen_movies = original_ratings[original_ratings['userId'] == user_id]['movieId'].tolist()

    # Filtrar películas no vistas
    predictions = predicted_ratings_df.loc[user_id].drop(index=seen_movies)

    # Seleccionar las n mejores
    recommendations = predictions.sort_values(ascending=False)
    return recommendations

# Ejemplo de recomendaciones para un usuario
user_id = 547
recommendations = recommend_movies(user_id, predicted_ratings_df, train_data, n=5)

ids = movie_finder(list(recommendations.keys()), movies, 5)
    
recommendations_print = {id: recommendations[id] for id in ids}
print_recommendations(recommendations_print, user_id)

metadata_extractor(ids, movies)