import pandas as pd 
import numpy as np
from extreure_metadata import metadata_extractor, movie_finder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# def recommend_movies(active_user, user_movie_matrix, user_similarity_df, n_recommendations=5, n_similar_users=20):
#     # Usuaris similars
#     # similar_users = user_similarity_df[active_user].sort_values(ascending=False).index[1:n_similar_users+1]

#     # Predicció de puntuacions
#     recommendations = predict_ratings(active_user, user_movie_matrix, user_similarity_df, n_similar_users)
    
#     return recommendations

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

def update_user_similarity(active_user, user_movie_matrix, user_similarity_df):
    """
    Actualitza la fila i la columna corresponents a l'usuari actiu
    en la matriu de similitud.
    """
    for user in user_movie_matrix.index:
        if user == active_user:
            user_similarity_df.loc[active_user, user] = 1.0  # Similaritat amb si mateix
        else:
            vector_active = user_movie_matrix.loc[active_user].values
            vector_user = user_movie_matrix.loc[user].values

            mask = (vector_active != 0) & (vector_user != 0)

            if not np.any(mask):
                similarity = 0.0
            else:
                active_filtered = vector_active[mask]
                user_filtered = vector_user[mask]

                numerator = np.dot(active_filtered, user_filtered)
                denominator = np.linalg.norm(active_filtered) * np.linalg.norm(user_filtered)
                similarity = numerator / denominator if denominator != 0 else 0.0

            user_similarity_df.loc[active_user, user] = similarity
            user_similarity_df.loc[user, active_user] = similarity
        
def predict_ratings(active_user, user_movie_matrix, user_similarity_df, n_similar_users):
    # Mitja de les puntuacions de l'usuari actiu
    user_mean_rating = user_movie_matrix.loc[active_user][user_movie_matrix.loc[active_user] != 0].mean()
    calificaciones_usuario1 = user_movie_matrix.loc[active_user]
    num_films_usuario1 = (user_movie_matrix.loc[active_user] != 0).sum()
    varianzas = user_movie_matrix.var()
    maxima = varianzas.max()
    minima = varianzas.min()

    # Inicialitzar les prediccions
    predicted_ratings = {}

    for movie in user_movie_matrix.columns:
        if user_movie_matrix.loc[active_user, movie] == 0: # Només predim les pel·lícules no vistes
            numerator = 0
            denominator = 0
            variacion = varianzas[movie]
            variacion_norm = np.log1p(variacion - minima + 1e-5) / np.log1p(maxima - minima + 1e-5)

            posible_users = user_similarity_df[active_user].sort_values(ascending=False)
            similar_users = []
            for similar_user in posible_users.keys():
                if user_movie_matrix.loc[similar_user, movie] != 0 and similar_user != active_user:
                    similar_users.append(similar_user)
                if len(similar_users) >= n_similar_users:
                    break

            for similar_user in similar_users:
                calificaciones_usuario2 = user_movie_matrix.loc[similar_user]
                films_comun = (calificaciones_usuario1 != 0) & (calificaciones_usuario2 != 0)
                num_films_usuario2 = (user_movie_matrix.loc[similar_user] != 0).sum()
                num_films_en_comun = films_comun.sum()  # Suma de True (1)
                jaccard = (num_films_en_comun/ (num_films_usuario1 + num_films_usuario2 - num_films_en_comun))

                # Calcular la diferència centrada
                rating_difference = user_movie_matrix.loc[similar_user, movie] - user_movie_matrix.loc[similar_user][user_movie_matrix.loc[similar_user] != 0].mean()
                
                # Ponderar per similitud
                similarity = user_similarity_df.loc[active_user, similar_user]
                numerator += similarity * rating_difference * jaccard * variacion_norm
                denominator += similarity * jaccard * variacion_norm

            # Calcular la predicció si el denominador no es zero
            if denominator > 0:
                predicted_ratings[movie] = user_mean_rating + (numerator / denominator)

    # Convertir a DataFrame
    predicted_ratings_df = pd.Series(predicted_ratings)

    # Normalitzar les dades
    min_rating = predicted_ratings_df.min()
    max_rating = predicted_ratings_df.max()
    predicted_ratings_df = 0 + 5 * (predicted_ratings_df - min_rating) / (max_rating - min_rating)

    return predicted_ratings_df.sort_values(ascending=False)

def test(active_user, user_movie_matrix, user_similarity_df, n_similar_users):
    # Mitja de les puntuacions de l'usuari actiu
    user_mean_rating = user_movie_matrix.loc[active_user][user_movie_matrix.loc[active_user] != 0].mean()
    calificaciones_usuario1 = user_movie_matrix.loc[active_user]
    num_films_usuario1 = (user_movie_matrix.loc[active_user] != 0).sum()
    varianzas = user_movie_matrix.var()
    maxima = varianzas.max()
    minima = varianzas.min()

    # Inicialitzar les prediccions
    predicted_ratings = {}

    for movie in user_movie_matrix.columns:
        numerator = 0
        denominator = 0
        variacion = varianzas[movie]
        variacion_norm = np.log1p(variacion - minima + 1e-5) / np.log1p(maxima - minima + 1e-5)

        posible_users = user_similarity_df[active_user].sort_values(ascending=False)
        similar_users = []
        for similar_user in posible_users.keys():
            if user_movie_matrix.loc[similar_user, movie] != 0 and similar_user != active_user:
                similar_users.append(similar_user)
            if len(similar_users) >= n_similar_users:
                break

        for similar_user in similar_users:
            calificaciones_usuario2 = user_movie_matrix.loc[similar_user]
            films_comun = (calificaciones_usuario1 != 0) & (calificaciones_usuario2 != 0)
            num_films_usuario2 = (user_movie_matrix.loc[similar_user] != 0).sum()
            num_films_en_comun = films_comun.sum()  # Suma de True (1)
            jaccard = (num_films_en_comun/ (num_films_usuario1 + num_films_usuario2 - num_films_en_comun))

            # Calcular la diferència centrada
            rating_difference = user_movie_matrix.loc[similar_user, movie] - user_movie_matrix.loc[similar_user][user_movie_matrix.loc[similar_user] != 0].mean()
            
            # Ponderar per similitud
            similarity = user_similarity_df.loc[active_user, similar_user]
            numerator += similarity * rating_difference * jaccard * variacion_norm
            denominator += similarity * jaccard * variacion_norm

        # Calcular la predicció si el denominador no es zero
        if denominator > 0:
            predicted_ratings[movie] = user_mean_rating + (numerator / denominator)

    # Convertir a DataFrame
    predicted_ratings_df = pd.Series(predicted_ratings)

    # Normalitzar les dades
    # min_rating = predicted_ratings_df.min()
    # max_rating = predicted_ratings_df.max()
    # predicted_ratings_df = 0 + 5 * (predicted_ratings_df - min_rating) / (max_rating - min_rating)

    return predicted_ratings_df.sort_values(ascending=False)

# def predict_single_movie(active_user, movie_id, user_movie_matrix, user_similarity_df, N = 20):
#     """
#     Prediu la puntuació d'una pel·lícula específica per a un usuari, 
#     assumint que no l'ha vist.
#     """
#     # Crear una còpia de la matriu de puntuacions
#     modified_matrix = user_movie_matrix.copy()
    
#     # Eliminar la puntuació de la pel·lícula per a l'usuari actiu
#     modified_matrix.loc[active_user, movie_id] = 0

#     # Crear una còpia local de la matriu de similitud
#     local_similarity_df = user_similarity_df.copy()

#     # Actualitzar la matriu de similitud només per a l'usuari actiu
#     update_user_similarity(active_user, modified_matrix, local_similarity_df)

#     # Usuaris similars
#     posible_users = local_similarity_df[active_user].sort_values(ascending=False)
#     similar_users = []
#     for similar_user in posible_users.keys():
#         if user_movie_matrix.loc[similar_user, movie_id] != 0 and similar_user != active_user:
#             similar_users.append(similar_user)
#         if len(similar_users) >= N:
#             break

#     # Predicció per a la pel·lícula específica
#     user_mean_rating = user_movie_matrix.loc[active_user][user_movie_matrix.loc[active_user] != 0].mean()
#     numerator = 0
#     denominator = 0
#     calificaciones_usuario1 = user_movie_matrix.loc[active_user]
#     num_films_usuario1 = (user_movie_matrix.loc[active_user] != 0).sum()
#     print(num_films_usuario1)
#     varianzas = user_movie_matrix.var()
#     maxima = varianzas.max()
#     minima = varianzas.min()
#     variacion = varianzas[movie_id]
#     variacion_norm = np.log1p(variacion - minima + 1e-5) / np.log1p(maxima - minima + 1e-5)

#     for similar_user in similar_users: 
#         calificaciones_usuario2 = user_movie_matrix.loc[similar_user]
#         films_comun = (calificaciones_usuario1 != 0) & (calificaciones_usuario2 != 0)
#         num_films_usuario2 = (user_movie_matrix.loc[similar_user] != 0).sum()
#         num_films_en_comun = films_comun.sum()  # Suma de True (1)
        
#         rating_difference = user_movie_matrix.loc[similar_user, movie_id] - user_movie_matrix.loc[similar_user][user_movie_matrix.loc[active_user] != 0].mean()
#         # rating_difference = user_movie_matrix.loc[similar_user, movie_id] - user_mean_rating
#         similarity = local_similarity_df.loc[active_user, similar_user] 
#         jaccard = (num_films_en_comun/ (num_films_usuario1 + num_films_usuario2 - num_films_en_comun))
#         numerator += similarity * rating_difference * jaccard * variacion_norm
#         denominator += similarity * jaccard * variacion_norm

#     if denominator > 0:
#         print("\nValue:", (numerator / denominator))
#         print("mean:", user_mean_rating)
#         predicted_rating = user_mean_rating + (numerator / denominator)
#         return user_mean_rating + (numerator / denominator)  # Normalitzar entre 1 i 5

#     return None  # Si no es pot predir

def predict_single_movie(user_id, movie_id, user_movie_matrix, user_similarity_matrix, variance_weights):
    user_ratings = user_movie_matrix.loc[user_id]
    valid_users = user_movie_matrix.index[user_movie_matrix[movie_id].notna() & (user_movie_matrix.index != user_id)]
    
    # Validate indices
    valid_users = valid_users.intersection(variance_weights.index)

    if len(valid_users) == 0:
        return np.nan  # No valid users for prediction

    similarities = user_similarity_matrix[user_id][valid_users]
    weights = similarities * variance_weights.loc[valid_users]

    ratings = user_movie_matrix.loc[valid_users, movie_id]
    weighted_sum = (weights * ratings).sum()
    normalization_factor = weights.sum() + 1e-8  # Avoid division by zero

    return weighted_sum / normalization_factor

def print_recommendations(recommendations, userId):
    print(f"\nRecomanacions de pel·lícules per l'usuari amb ID: {userId}:\n")
    for i, (title, score) in enumerate(recommendations.items()):
        print(f"{i+1}. {title} (Puntuació predita: {score:.2f})")
    print()

def save_similarity_matrix_to_csv(similarity_matrix, file_path):
    similarity_matrix.to_csv(file_path)

def load_similarity_matrix_from_csv(file_path):
    return pd.read_csv(file_path, index_col=0)

def evaluate_model(user_movie_matrix, user_similarity_matrix, variance_weights):
    actual_ratings = []
    predicted_ratings = []

    for user_id in user_movie_matrix.index:
        for movie_id in user_movie_matrix.columns:
            actual_rating = user_movie_matrix.loc[user_id, movie_id]
            if not np.isnan(actual_rating):
                predicted_rating = predict_single_movie(
                    user_id, movie_id, user_movie_matrix, user_similarity_matrix, variance_weights
                )
                actual_ratings.append(actual_rating)
                predicted_ratings.append(predicted_rating)

    # Calculate MAE and MSE
    actual_ratings = np.array(actual_ratings)
    predicted_ratings = np.array(predicted_ratings)
    mae = np.mean(np.abs(actual_ratings - predicted_ratings))
    mse = np.mean((actual_ratings - predicted_ratings) ** 2)

    print(f"Model Evaluation:\nMAE (Mean Absolute Error): {mae:.4f}\nMSE (Mean Squared Error): {mse:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.hist([actual_ratings, predicted_ratings], label=['Actual', 'Predicted'], bins=20, alpha=0.7)
    plt.legend()
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Actual vs Predicted Ratings')
    plt.show()

def compute_variance_weights(user_movie_matrix):
    return user_movie_matrix.var(axis=0).apply(np.log1p)

def compute_user_similarity(user_movie_matrix):
    similarity_matrix = 1 - cosine(user_movie_matrix.T)
    return similarity_matrix

def recommend_movies(user_id, user_movie_matrix, user_similarity_matrix, variance_weights, n_recommendations=5):
    user_ratings = user_movie_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings.isna()].index

    predictions = {}
    for movie_id in unrated_movies:
        predictions[movie_id] = predict_single_movie(
            user_id, movie_id, user_movie_matrix, user_similarity_matrix, variance_weights
        )

    # Sort movies by predicted ratings
    recommended_movies = pd.Series(predictions).sort_values(ascending=False).head(n_recommendations)
    return recommended_movies

if __name__ == "__main__":
    # Inicialització dataframes
    ratings = pd.read_csv('./Data/ratings_small.csv') # movieid = int64
    movies_long = pd.read_csv('./Data/movies_metadata.csv', low_memory=False) # id = str

    # # Agafem només les columnes que interessen i unifiquem tant el nom com la unitat
    # movies = movies_long[['id', 'title']].rename(columns={'id': 'movieId'})

    # movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
    # movies = movies.dropna(subset=['movieId'])
    # movies['movieId'] = movies['movieId'].astype(int)

    # ratings_with_titles = ratings.merge(movies, on='movieId', how='inner')

    # Creem la matriu on les files son els usuaris i les columnes son les pel·lícules
    user_movie_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
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

    active_user = 195

    # # Una vegada passem les dades a la matriu de puntuacions User-Item i seleccionem l'usuari, podem aplicar el model
    # recommendations = recommend_movies(active_user, user_movie_matrix_filled, user_similarity_df)

    # # Comprovar que les recomanacions es troben a la database de metadata, i després imprimir-les
    # ids = movie_finder(list(recommendations.keys()), movies_long, 5)
    
    # recommendations_print = {id: recommendations[id] for id in ids}
    # print_recommendations(recommendations_print, active_user)
    
    # metadata_extractor(ids, movies_long)

    # # Predir una pel·lícula específica
    # movie_to_predict = 111
    # print(f"\nAlgorisme de predicció de puntuació de l'usuari {active_user} a la pel·lícula {movie_to_predict}:")
    # # id_predict = movie_finder([movie_to_predict], movies_long, 5)
    # # metadata_extractor(id_predict, movies_long)
    
    # predicted_score = predict_single_movie(active_user, movie_to_predict, user_movie_matrix_filled, user_similarity_df, 20)
    # if predicted_score:
    #     print(f"\nPuntuació predita per l'usuari amb ID {active_user} a la pel·lícula amb ID {movie_to_predict}: {predicted_score:.2f}")
    #     print(f"Puntuació original: {user_movie_matrix.loc[active_user, movie_to_predict]}")
    # else:
    #     print(f"\nNo s'ha pogut predir una puntuació per a la pel·lícula amb ID {movie_to_predict}.")


    # df = pd.read_csv('./Data/users_small.csv')
    # pelis = eval(df[df['userId'] == active_user]['ratings'].values[0])
    # scores = {}
    # for peli in pelis:
    #     movie_to_predict = peli['movieId']
    #     # print(f"\nAlgorisme de predicció de puntuació de l'usuari {active_user} a la pel·lícula {movie_to_predict}:")
    #     # id_predict = movie_finder([movie_to_predict], movies_long, 5)
    #     # metadata_extractor(id_predict, movies_long)
        
    #     predicted_score = predict_single_movie(active_user, movie_to_predict, user_movie_matrix_filled, user_similarity_df, 20)
    #     if predicted_score:
    #         scores[movie_to_predict] = predicted_score
    #     #     print(f"\nPuntuació predita per l'usuari amb ID {active_user} a la pel·lícula amb ID {movie_to_predict}: {predicted_score:.2f}")
    #     #     print(f"Puntuació original: {user_movie_matrix.loc[active_user, movie_to_predict]}")
    #     # else:
    #     #     print(f"\nNo s'ha pogut predir una puntuació per a la pel·lícula amb ID {movie_to_predict}.")
    # print(min(scores.values()))
    # print(max(scores.values()))
    
    # recommendations = test(active_user, user_movie_matrix_filled, user_similarity_df, 20)

    # df = pd.read_csv('./Data/users_small.csv')
    # pelis = eval(df[df['userId'] == active_user]['ratings'].values[0])

    # for peli in pelis:
    #     try:
    #         print(f"\nPelicula {peli['movieId']}:")
    #         print(f"    - Puntuació usuari {active_user}: {peli['rating']}")
    #         print(f"    - Predicció: {recommendations[peli['movieId']]}")
    #     except:
    #         pass

    user_similarity_matrix = compute_user_similarity(user_movie_matrix)
    variance_weights = compute_variance_weights(user_movie_matrix)

    evaluate_model(user_movie_matrix, user_similarity_matrix, variance_weights)

    # Get recommendations for a user
    recommendations = recommend_movies("User1", user_movie_matrix, user_similarity_matrix, variance_weights)
    print("Recommendations for User1:")
    print(recommendations)
