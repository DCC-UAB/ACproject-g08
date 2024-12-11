import pandas as pd
import numpy as np

# Calcular la Adjusted Cosine Similarity
def adjusted_cosine_similarity(matrix):
    # Calcular la mitja per usuari (files)
    user_means = matrix.mean(axis=1)
    
    # Centrar els ratings per usuari
    centered_matrix = matrix.sub(user_means, axis=0)
    
    # Calcular similitud ajustada cosinus entre pel·lícules (columnes)
    similarity = np.dot(centered_matrix.T, centered_matrix)
    
    # Calcular les normes de les columnes
    norms = np.sqrt(np.sum(centered_matrix**2, axis=0))  # Normas por columna
    norms_matrix = np.outer(norms, norms)
    
    # Similitud ajustada
    adjusted_similarity = similarity / (norms_matrix)
    
    return pd.DataFrame(adjusted_similarity, 
                        index=matrix.columns, 
                        columns=matrix.columns)

# Función para predecir ratings de un usuario
def predict_ratings(user_id, user_movie_matrix, item_similarity):
    user_ratings = user_movie_matrix.loc[user_id]  # Calificaciones del usuario
    rated_movies = user_ratings.dropna()  # Películas que ha calificado el usuario

    predicted_ratings = {}

    # Inicializar los sumatorios
    weighted_sum = 0
    similarity_sum = 0

    for movie in rated_movies.index:  # Solo considerar las películas que el usuario ha calificado
        similarity = item_similarity[movie]  # Similitud de la película con todas las demás
        similarity_rated_movies = similarity[rated_movies.index]  # Filtramos las similitudes solo para las películas calificadas
        
        # Multiplicación de similitudes por las calificaciones
        weighted_sum = np.dot(similarity_rated_movies, rated_movies)
        # Suma de las similitudes
        similarity_sum = similarity_rated_movies.sum()

        # Evitar división por cero
        if similarity_sum != 0:
            predicted_ratings[movie] = weighted_sum / similarity_sum
        else:
            predicted_ratings[movie] = 0  # Si no hay similitud, asignar 0
    
    return pd.Series(predicted_ratings)

if __name__ == "__main__":
    # Inicialización de los DataFrames
    ratings = pd.read_csv('./Data/ratings_small.csv')  # movieId = int64
    movies = pd.read_csv('./Data/movies_metadata.csv', low_memory=False)  # id = str

    # Filtrar y unificar columnas relevantes
    movies = movies[['id', 'title']].rename(columns={'id': 'movieId'})
    movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
    movies = movies.dropna(subset=['movieId'])
    movies['movieId'] = movies['movieId'].astype(int)

    # Combinar ratings y títulos
    ratings_with_titles = ratings.merge(movies, on='movieId', how='inner')

    # Crear la matriz usuario-película
    user_movie_matrix = ratings_with_titles.pivot_table(
        index='userId',
        columns='title',
        values='rating'
    )

    # Rellenar valores NaN con ceros
    user_movie_matrix_filled = user_movie_matrix.fillna(0)

    # Calcular la similitud entre películas
    item_similarity = adjusted_cosine_similarity(user_movie_matrix_filled)

    # Obtener predicciones para un usuario
    user_id = 2
    predicted_ratings = predict_ratings(user_id, user_movie_matrix, item_similarity)

    # Recomendar las mejores películas
    top_recommendations = predicted_ratings.sort_values(ascending=False).tail(5)
    print("Recomendaciones para el usuario", user_id)
    print(top_recommendations)