import content_based_cleaner
import numpy as np
import pandas as pd
import colaborative_cleaner
import graph
import time

def calculate_user_similarity_matrix(user_movie_matrix):
    """
    Calcula la matriu de similitut Pearson entre els usuaris (només es tenen en compte
    les pel·lícules que ambdós usuaris han puntuat).

    La funció rep una user_movie_matrix: DataFrame amb usuaris como filas
    i items com columnes.

    Retorna un DataFrame: matriu de similitut entre usuaris.
    """
    users = user_movie_matrix.index
    similarity_matrix = pd.DataFrame(index=users, columns=users, dtype=float)

    for user1 in users:
        user1_mean_rating = user_movie_matrix.loc[user1][user_movie_matrix.loc[user1] != 0].mean()
        for user2 in users:
            if user1 == user2:
                similarity_matrix.loc[user1, user2] = 1.0  # Similaridad con uno mismo es 1
            else:
                user2_mean_rating = user_movie_matrix.loc[user2][user_movie_matrix.loc[user2] != 0].mean()
                vector_user1 = user_movie_matrix.loc[user1].values
                vector_user2 = user_movie_matrix.loc[user2].values

                mask = (vector_user1 != 0) & (vector_user2 != 0)
    
                # Si no hi ha items comuns puntuats, retornem similitud 0
                if not np.any(mask):
                    similarity_matrix.loc[user1, user2] = 0.0
                    
                else:
                    # Seleccionar només els items puntuats per ambdós usuaris
                    user1_filtered = vector_user1[mask]
                    user2_filtered = vector_user2[mask]

                    user1_filtered = user1_filtered - np.array([user1_mean_rating] * len(user1_filtered))
                    user2_filtered = user2_filtered - np.array([user2_mean_rating] * len(user2_filtered))
                    
                    # Calcular similitut cosinus
                    numerator = np.dot(user1_filtered, user2_filtered)
                    denominator = np.linalg.norm(user1_filtered) * np.linalg.norm(user2_filtered)
                    
                    if denominator != 0:
                        similarity_matrix.loc[user1, user2] = numerator / denominator 
                    else:
                        similarity_matrix.loc[user1, user2] = 0.0

    return similarity_matrix

def create_similarity():
    ratings = pd.read_csv('./Data/ratings_small.csv') # movieid = int64
    user_movie_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    )
    user_movie_matrix_filled = user_movie_matrix.fillna(0)
    user_similarity_matrix = calculate_user_similarity_matrix(user_movie_matrix_filled)
    similarity_matrix_path = './Data/user_similarity_matrix.csv'
    user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_movie_matrix_filled.index, columns=user_movie_matrix_filled.index)
    user_similarity_df.to_csv(similarity_matrix_path)

def calculate_jaccard_similarity_matrix(user_movie_matrix):
    binary_matrix = (user_movie_matrix != 0).astype(int)
    jaccard_matrix = np.zeros((binary_matrix.shape[0], binary_matrix.shape[0]))

    for i, user_i in enumerate(binary_matrix.index):
        for j, user_j in enumerate(binary_matrix.index):
            if i <= j:  # Compute only for upper triangular and diagonal
                intersection = (binary_matrix.loc[user_i] & binary_matrix.loc[user_j]).sum()
                union = (binary_matrix.loc[user_i] | binary_matrix.loc[user_j]).sum()
                jaccard_matrix[i, j] = jaccard_matrix[j, i] = intersection / union if union > 0 else 0

    return pd.DataFrame(jaccard_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def create_jaccard():
    ratings = pd.read_csv('./Data/ratings_small.csv') # movieid = int64
    user_movie_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    )
    user_movie_matrix_filled = user_movie_matrix.fillna(0)
    user_similarity_matrix = calculate_jaccard_similarity_matrix(user_movie_matrix_filled)
    time.sleep(5)
    similarity_matrix_path = './Data/jaccard_similarity_matrix.csv'
    user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_movie_matrix_filled.index, columns=user_movie_matrix_filled.index)
    user_similarity_df.to_csv(similarity_matrix_path)

if __name__ == '__main__':
    # content_based_cleaner.afegir_generes()
    # content_based_cleaner.clean_empty_keywords()
    # content_based_cleaner.clean_ratings()
    # create_similarity()
    create_jaccard()
    # colaborative_cleaner.user_cleaner()
    # colaborative_cleaner.item_cleaner()
    # graph.extreure_users()
    # graph.extreure_movies()
    # graph.main()
