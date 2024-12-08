import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def recommend_movies(active_user, user_movie_matrix, user_similarity_df, n_recommendations=5):
    # Usuaris similars
    similar_users = user_similarity_df[active_user].sort_values(ascending=False).index[1:]

    # Predicció de puntuacions
    recommendations = predict_ratings_advanced(active_user, user_movie_matrix, user_similarity_df, similar_users)
    
    return recommendations.head(n_recommendations)

def predict_ratings_advanced(active_user, user_movie_matrix, user_similarity_df, similar_users):
    # Mitja de les puntuacions de l'usuari actiu
    user_mean_rating = user_movie_matrix.loc[active_user].mean()

    # Inicialitzar las prediccions
    predicted_ratings = {}

    for movie in user_movie_matrix.columns:
        if pd.isna(user_movie_matrix.loc[active_user, movie]):  # Només predim les pel·lícules no vistes
            numerator = 0
            denominator = 0

            for similar_user in similar_users:
                if not pd.isna(user_movie_matrix.loc[similar_user, movie]):  # Usuari similar ha puntuat la película
                    # Calcular la diferència centrada
                    rating_difference = user_movie_matrix.loc[similar_user, movie] - user_movie_matrix.loc[similar_user].mean()
                    
                    # Ponderar per similitud
                    similarity = user_similarity_df.loc[active_user, similar_user]
                    numerator += similarity * rating_difference
                    denominator += abs(similarity)

            # Calcular la predicció si el denominador no es zero
            if denominator > 0:
                predicted_ratings[movie] = user_mean_rating + (numerator / denominator)

    # Per normalitzar les dades entre valors de 1 a 5, primerament creem un scaler
    scaler = MinMaxScaler(feature_range=(1, 5))

    # Convertim les prediccions a DataFrame de Pandas
    predictions_df = pd.Series(predicted_ratings)

    # Normalitzem les prediccions amb el MinMaxScaler
    predictions_normalized = scaler.fit_transform(predictions_df.values.reshape(-1, 1))

    # Convertim les prediccions normalitzades un altre cop a una Serie 
    predictions_normalized_df = pd.Series(predictions_normalized.flatten(), index=predictions_df.index)
                
    return predictions_normalized_df.sort_values(ascending=False)

def print_recommendations_tabulated(recommendations, userId):
    max_title_length = max(len(title) + 1 for title in recommendations.index)  # Trobaa la longitud màxima dels títols
    msgtitol = "Títol"
    msgmean = "Puntació mitja dels usuaris similars"

    print(f"Recomanacions per l'usuari amb id: {userId}:\n")
    print(f"{msgtitol.ljust(max_title_length)} {msgmean}\n")
    
    for title, mean in recommendations.items():
        print(f"{title.ljust(max_title_length)} {mean}")

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

    # Creem la matriu on les files osn els usuaris i les columnes son les pel·lícules
    user_movie_matrix = ratings_with_titles.pivot_table(
        index='userId',
        columns='title',
        values='rating'
    )

    user_movie_matrix_filled = user_movie_matrix.fillna(0) # Els valors NaN es transformen a 0 per poder operar amb ells

    # Calcular similitud entre usuaris i crear el dataframe corresponent
    user_similarity = cosine_similarity(user_movie_matrix_filled)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    active_user = 2 # Cambiar per l'usuari desitjat

    # Una vegada passem les dades a la matriu de puntuacions User-Item i seleccionem l'usuari, podem aplicar el model
    recommendations = recommend_movies(active_user, user_movie_matrix, user_similarity_df)

    # Imprimir les recomanacions
    print_recommendations_tabulated(recommendations, active_user)