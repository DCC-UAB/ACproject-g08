import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix

# Cargar los datos
ratings = pd.read_csv('./Data/ratings.csv')

# Dividir en entrenamiento (80%) y prueba (20%)
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# # Guardar las divisiones (opcional)
# train_data.to_csv('./Data/train.csv', index=False)
# test_data.to_csv('./Data/test.csv', index=False)

# Crear una matriz dispersa usuario-película
user_movie_matrix_sparse = csr_matrix(
    (train_data['rating'], (train_data['userId'], train_data['movieId']))
)

rmse_values = []
k_values = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 50, 100, 150, 200, 300, 500]
# k_values = [10] # valor òptim pel model: 3.0098

for k in k_values:
    U, sigma, Vt = svds(user_movie_matrix_sparse, k=k)
    sigma = np.diag(sigma)

    block_size = 20  # Número de usuarios por bloque
    num_users = user_movie_matrix_sparse.shape[0]

    predicted_ratings = []

    for i in range(0, num_users, block_size):
        user_block = user_movie_matrix_sparse[i:i+block_size, :]
        U, sigma, Vt = svds(user_block, k=k)
        sigma = np.diag(sigma)

        U = U.astype(np.float32)
        sigma = sigma.astype(np.float32)
        Vt = Vt.astype(np.float32)

        block_ratings = np.dot(np.dot(U, sigma), Vt)
        predicted_ratings.append(block_ratings)

    # Combinar las predicciones de los bloques
    predicted_ratings = np.vstack(predicted_ratings)
    
    # predicted_ratings = np.dot(np.dot(U, sigma), Vt)

    predicted_ratings_df = pd.DataFrame(
        predicted_ratings, 
        index=np.arange(user_movie_matrix_sparse.shape[0]),  # Asignar índices de usuarios
        columns=np.arange(user_movie_matrix_sparse.shape[1])  # Asignar índices de películas
    )

    # predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_movie_matrix.index, columns=user_movie_matrix.columns)

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

    # Filtrar las películas vistas para que solo incluyan las que están en las columnas de la matriz
    seen_movies = [movie for movie in seen_movies if movie in predicted_ratings_df.columns]

    # Filtrar películas no vistas
    predictions = predicted_ratings_df.loc[user_id].drop(index=seen_movies)

    # Seleccionar las n mejores
    recommendations = predictions.sort_values(ascending=False).head(n)
    return recommendations

# Ejemplo de recomendaciones para un usuario
user_id = 2
recommendations = recommend_movies(user_id, predicted_ratings_df, train_data, n=5)
print(recommendations)