import pandas as pd 
import numpy as np

def recommend_movies(active_user, user_movie_matrix, user_similarity_df, n_recommendations=5, n_similar_users=100):
    # Usuaris similars
    similar_users = user_similarity_df[active_user].sort_values(ascending=False).index[1:n_similar_users+1]

    # Predicció de puntuacions
    recommendations = predict_ratings(active_user, user_movie_matrix, user_similarity_df, similar_users)
    
    return recommendations.head(n_recommendations)

def calculate_user_similarity_matrix(user_movie_matrix):
    """
    Calcula la matriu de similitut cosinus entre els usuaris (només es tenen en compte
    les pel·lícules que ambdós usuaris han puntuat).

    La funció rep una user_movie_matrix: DataFrame amb usuaris como filas
    i items com columnes.

    Retorna un DataFrame: matriu de similitut entre usuaris.
    """
    users = user_movie_matrix.index
    similarity_matrix = pd.DataFrame(index=users, columns=users, dtype=float)

    for user1 in users:
        for user2 in users:
            if user1 == user2:
                similarity_matrix.loc[user1, user2] = 1.0  # Similaridad con uno mismo es 1
            else:
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
                    
                    # Calcular similitut cosinus
                    numerator = np.dot(user1_filtered, user2_filtered)
                    denominator = np.linalg.norm(user1_filtered) * np.linalg.norm(user2_filtered)
                    
                    if denominator != 0:
                        similarity_matrix.loc[user1, user2] = numerator / denominator 
                    else:
                        similarity_matrix.loc[user1, user2] = 0.0

    return similarity_matrix
        
def predict_ratings(active_user, user_movie_matrix, user_similarity_df, similar_users):
    # Mitja de les puntuacions de l'usuari actiu
    user_mean_rating = user_movie_matrix.loc[active_user].mean()

    # Inicialitzar les prediccions
    predicted_ratings = {}

    for movie in user_movie_matrix.columns:
        if user_movie_matrix.loc[active_user, movie] == 0:  # Només predim les pel·lícules no vistes
            numerator = 0
            denominator = 0

            for similar_user in similar_users:
                if user_movie_matrix.loc[similar_user, movie] != 0:  # Usuari similar ha puntuat la película
                    # Calcular la diferència centrada
                    rating_difference = user_movie_matrix.loc[similar_user, movie] - user_movie_matrix.loc[similar_user].mean()
                    
                    # Ponderar per similitud
                    similarity = user_similarity_df.loc[active_user, similar_user]
                    numerator += similarity * rating_difference
                    denominator += similarity

            # Calcular la predicció si el denominador no es zero
            if denominator > 0:
                predicted_ratings[movie] = user_mean_rating + (numerator / denominator)

    # Convertir a DataFrame
    predicted_ratings_df = pd.Series(predicted_ratings)

    # Normalitzar les dades
    min_rating = predicted_ratings_df.min()
    max_rating = predicted_ratings_df.max()
    predicted_ratings_df = 1 + 4 * (predicted_ratings_df - min_rating) / (max_rating - min_rating)

    return predicted_ratings_df.sort_values(ascending=False)

def print_recommendations(recommendations, userId):
    print(f"Recomanacions de pel·lícules per l'usuari amb ID: {userId}:\n")
    for i, (title, score) in enumerate(recommendations.items()):
        print(f"{i+1}. {title} (Puntuació predita: {score:.2f})")
    print()

def save_similarity_matrix_to_csv(similarity_matrix, file_path):
    similarity_matrix.to_csv(file_path)

def load_similarity_matrix_from_csv(file_path):
    return pd.read_csv(file_path, index_col=0)

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

    # Creem la matriu on les files son els usuaris i les columnes son les pel·lícules
    user_movie_matrix = ratings_with_titles.pivot_table(
        index='userId',
        columns='title',
        values='rating'
    )
    user_movie_matrix_filled = user_movie_matrix.fillna(0)
    similarity_matrix_path = './Data/user_similarity_matrix.csv'

    try:
        # Intentar carregar la matriu de similitud des de l'arxiu
        user_similarity_df = load_similarity_matrix_from_csv(similarity_matrix_path)
        user_similarity_df.index = user_similarity_df.index.astype(int)
        user_similarity_df.columns = user_similarity_df.columns.astype(int)

    except FileNotFoundError:
        # Si no existeix l'arxiu, calcular la matriu de similitud i guardar-la
        user_similarity_matrix = calculate_user_similarity_matrix(user_movie_matrix_filled)
        user_similarity_df = pd.DataFrame(user_similarity_matrix, index=user_movie_matrix_filled.index, columns=user_movie_matrix_filled.index)
        save_similarity_matrix_to_csv(user_similarity_df, similarity_matrix_path)

    active_user = 2 # Cambiar per l'usuari desitjat

    # Una vegada passem les dades a la matriu de puntuacions User-Item i seleccionem l'usuari, podem aplicar el model
    recommendations = recommend_movies(active_user, user_movie_matrix_filled, user_similarity_df)

    # Imprimir les recomanacions
    print_recommendations(recommendations, active_user)