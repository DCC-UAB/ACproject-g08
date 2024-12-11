import pandas as pd
import numpy as np

# Calcular la Adjusted Cosine Similarity
def calculate_item_similarity_matrix(user_movie_matrix):
    """
    Calcula la matriu de similitud ajustada cosinus entre ítems (pel·lícules).

    La similitud es basa en les puntuacions centrades respecte a la mitjana de l'usuari.
    Es considera que si un usuari no ha puntuat una pel·lícula, la puntuació centrada és 0.

    Args:
    - user_movie_matrix: DataFrame amb usuaris com a files i pel·lícules com a columnes.

    Retorna:
    - DataFrame: matriu de similitud entre ítems.
    """
    # Centrar les puntuacions de cada usuari respecte a la seva mitjana
    user_mean_ratings = user_movie_matrix.mean(axis=1)
    print(user_mean_ratings)
    print(user_movie_matrix)
    
    user_movie_matrix = user_movie_matrix.apply(lambda row: row.replace(0, row.mean()), axis=1)
    print(user_movie_matrix)

    mean_centered_matrix = user_movie_matrix.sub(user_mean_ratings, axis=0) 
    print(mean_centered_matrix) # Subtract user means

    # Transposar per calcular similitud entre ítems (columnes)
    items = mean_centered_matrix.columns
    similarity_matrix = pd.DataFrame(index=items, columns=items, dtype=float)

    for item1 in items:
        for item2 in items:
            if item1 == item2:
                similarity_matrix.loc[item1, item2] = 1.0  # Similaritat amb un mateix és 1
            else:
                vector_item1 = mean_centered_matrix[item1].values
                vector_item2 = mean_centered_matrix[item2].values
                    
                # Calcular similitut cosinus
                numerator = np.dot(vector_item1, vector_item2)
                denominator = np.linalg.norm(vector_item1) * np.linalg.norm(vector_item2)
                
                if denominator != 0:
                    similarity_matrix.loc[item1, item2] = numerator / denominator 
                else:
                    similarity_matrix.loc[item1, item2] = 0.0

    return similarity_matrix

# Función para predecir ratings de un usuario
def predict_ratings(user_id, user_movie_matrix, item_similarity):
    """
    Predice las puntuaciones de un usuario para las películas que no ha valorado.

    Args:
    - user_id: ID del usuario para el que se realizarán las predicciones.
    - user_movie_matrix: DataFrame con usuarios como filas y películas como columnas.
    - item_similarity: Matriz de similitud entre ítems.

    Retorna:
    - DataFrame: Predicciones ordenadas de mayor a menor puntuación.
    """
    user_ratings = user_movie_matrix.loc[user_id]
    predicted_ratings = {}

    for target_item in user_movie_matrix.columns:
        if user_ratings[target_item] == 0:  # Sólo predecir para ítems no valorados
            numerator = 0
            denominator = 0

            for rated_item in user_movie_matrix.columns:
                if user_ratings[rated_item] != 0:  # Sólo considerar ítems ya valorados
                    similarity = item_similarity.loc[target_item, rated_item]
                    numerator += similarity * user_ratings[rated_item]
                    denominator += similarity

            if denominator > 0:
                predicted_ratings[target_item] = numerator / denominator
            else:
                predicted_ratings[target_item] = 0  # Si no hay similitud, la predicción es 0

    # Convertir a Series y ordenar por puntuaciones
    predicted_ratings_series = pd.Series(predicted_ratings)
    
    return predicted_ratings_series.sort_values(ascending=False)

if __name__ == "__main__":
    # Inicialització dataframes
    ratings = pd.read_csv('./Data/ratings_small.csv') # movieid = int64
    movies = pd.read_csv('./Data/movies_metadata.csv', low_memory=False) # id = str

    # Agafem només les columnes que interessen i unifiquem tant el nom com la unitat
    movies = movies[['id', 'title']].rename(columns={'id': 'movieId'})

    movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
    movies = movies.dropna(subset=['movieId'])
    movies['movieId'] = movies['movieId'].astype(int)

    ratings_with_titles = ratings.merge(movies, on='movieId', how='inner')

    # # Creem la matriu on les files son els usuaris i les columnes son les pel·lícules
    # user_movie_matrix = ratings_with_titles.pivot_table(
    #     index='userId',
    #     columns='title',
    #     values='rating'
    # )
    # user_movie_matrix_filled = user_movie_matrix.fillna(0)

    matrix = [
    [2, 3, 5, 4, 0],
    [5, 0, 2, 2, 1],
    [1, 2, 5, 5, 5],
    [4, 5, 3, 0, 3],
    [1, 4, 1, 4, 1],
    [1, 2, 4, 3, 4],
    [4, 3, 1, 2, 1],
    [1, 1.5, 2.5, 2, 3],
    [2, 3, 4, 1, 0],
]

    # Convierte a DataFrame
    user_movie_matrix = pd.DataFrame(
        matrix,
        index=[i for i in range(len(matrix))],  
        columns=[f'Item {j+1}' for j in range(len(matrix[0]))]
    )
    user_movie_matrix_filled = user_movie_matrix.fillna(0)

    # Calcular la similitud entre películas
    item_similarity = calculate_item_similarity_matrix(user_movie_matrix_filled)

    print(item_similarity)

    # Obtener predicciones para un usuario
    user_id = 2
    predicted_ratings = predict_ratings(user_id, user_movie_matrix, item_similarity)

    # Recomendar las mejores películas
    top_recommendations = predicted_ratings.sort_values(ascending=False).tail(5)
    print("Recomendaciones para el usuario", user_id)
    print(top_recommendations)