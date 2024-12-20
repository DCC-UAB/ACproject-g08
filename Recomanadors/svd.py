import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

from Recomanadors.extreure_metadata import movie_finder, metadata_extractor
from Recomanadors.user_to_user_experiments import print_recommendations

# Funció per recomanar pel·lícules
def recommend_movies(user_id, predicted_ratings_df, original_ratings):
    seen_movies = original_ratings[original_ratings['userId'] == user_id]['movieId'].tolist()

    predictions = predicted_ratings_df.loc[user_id].drop(index=seen_movies)
    # predictions = predicted_ratings_df.loc[user_id]

    # Normalitzar les dades
    min_rating = predictions.min()
    max_rating = predictions.max()
    predictions = 1 + 4 * (predictions - min_rating) / (max_rating - min_rating)

    predictions = predictions = predictions.apply(custom_round)

    recommendations = predictions.sort_values(ascending=False)
    return recommendations

def custom_round(x):
        decimal_part = x - int(x)

        if decimal_part < 0.25:
            return int(x) 
        elif decimal_part < 0.75:
            return int(x) + 0.5
        else:
            return int(x) + 1

def calcular_svd(user_movie_matrix, user_movie_matrix_filled, ratings):
    rmse_values = []
    k_values = [175]

    for k in k_values:
        # Descomposició SVD
        U, sigma, Vt = svds(user_movie_matrix_filled, k=k)
        sigma = np.diag(sigma)

        # Reconstruir la matriu de prediccions
        predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        
        # Crear un DataFrame per a prediccions
        predicted_ratings_df = pd.DataFrame(
            predicted_ratings, 
            index=user_movie_matrix.index,
            columns=user_movie_matrix.columns)
        
        # Avaluar les prediccions
        predictions = []
        true_ratings = []
        for _, row in ratings.iterrows():
            user, movie, actual = row['userId'], row['movieId'], row['rating']
            pred = predicted_ratings_df.loc[user, movie]
            predictions.append(pred)
            true_ratings.append(actual)

        # Gestionar valors absents (p. ex., pel·lícules no predites)
        predictions = np.nan_to_num(predictions, nan=np.mean(true_ratings))

        # Calcular RMSE
        rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
        rmse_values.append((k, rmse))
        print(f"Algorisme SVD:")
        print(f"k = {k}, RMSE = {rmse:.4f}")
        print()

    return predicted_ratings_df

def main():
    # Carregar les dades
    ratings = pd.read_csv('./Data/ratings_small.csv')
    movies = pd.read_csv('./Data/movies_metadata.csv')

    # Crear la matriu usuari-pel·lícula
    user_movie_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating')

    # Omplir valors NaN amb la mitjana per usuari
    user_movie_matrix_filled = user_movie_matrix.apply(lambda row: row.fillna(row.mean()), axis=1).to_numpy()

    user_id = 547
    # Cas 1: Recomanar 5 pel·lícules a l'usuari 665
    print(f"Cas 1: Recomanacions per a l'usuari {user_id}")
    print()
    predicted_ratings_df = calcular_svd(user_movie_matrix, user_movie_matrix_filled, ratings)
    recommendations = recommend_movies(user_id, predicted_ratings_df, ratings)
    ids = movie_finder(list(recommendations.keys()), movies, 5)
    recommendations_print = {id: recommendations[id] for id in ids}
    print_recommendations(recommendations_print, user_id)
    metadata_extractor(ids, movies)

    # Cas 2: Excloure una pel·lícula i predir el seu valor
    movie_id_predict = 141
    print(f"Cas 2: Predir la valoració de la pel·lícula amb ID {movie_id_predict}")
    print()
    original_rating = user_movie_matrix.loc[user_id, movie_id_predict]
    user_ratings = user_movie_matrix.loc[user_id, :]
    mean_rating = user_ratings[user_ratings != 0].mean()

    # Substituir el valor a la posició
    user_movie_matrix.loc[user_id, movie_id_predict] = mean_rating 

    # Recalcular la matriu amb el valor eliminat
    user_movie_matrix_filled = user_movie_matrix.apply(lambda row: row.fillna(row.mean()), axis=1).to_numpy()
    predicted_ratings_df = calcular_svd(user_movie_matrix, user_movie_matrix_filled, ratings)

    # Obtenir la nova predicció per a la pel·lícula 32
    new_prediction = custom_round(predicted_ratings_df.loc[user_id, movie_id_predict])
    print(f"Valor original: {original_rating}, Predicció: {new_prediction}")
    print()

    # Mostrar metadades
    metadata_extractor([movie_id_predict], movies)

if __name__ == "__main__":
    main()
