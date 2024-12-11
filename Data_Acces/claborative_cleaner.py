import pandas as pd

# def matrix_convertion():
#     df = pd.read_csv('./Data/ratings.csv')

#     users = list(df['userId'].astype(int))
#     movies = list(df['movieId'].astype(int))
#     unique_movies = set(movies)
#     unique_movies = sorted(list(unique_movies))
#     ratings = list(df['userId'].astype(float))

#     rated = {}
#     for user, movie, rating in zip(users, movies, ratings):
#         try:
#             rated[user][movie] = rating
#         except:
#             rated[user] = {movie: rating}

def user_cleaner():
    # Cargar el archivo CSV
    df = pd.read_csv('./Data/ratings.csv')

    # Contar cuántos votos tiene cada película
    conteo_votos = df.groupby('movieId').size()

    # Filtrar películas que tienen exactamente un voto
    peliculas_unico_voto = conteo_votos[conteo_votos == 1].index

    # Obtener los usuarios que votaron por estas películas
    usuarios_unicos = df[df['movieId'].isin(peliculas_unico_voto)]

    # Identificar los usuarios que solo han votado por estas películas
    usuarios_con_unico_voto = usuarios_unicos['userId'].unique()

    # Filtrar los usuarios que no han votado por otras películas
    usuarios_finales = []
    for user in usuarios_con_unico_voto:
        if df[df['userId'] == user]['movieId'].isin(peliculas_unico_voto).count() == df[df['userId'] == user].shape[0]:
            usuarios_finales.append(user)

    # Filtrar el DataFrame para eliminar a estos usuarios
    df_filtrado = df[~df['userId'].isin(usuarios_finales)]

    # Guardar el nuevo DataFrame en un nuevo archivo CSV
    df_filtrado.to_csv('user_to_user_ratings.csv', index=False)

if __name__ == '__main__':
    user_cleaner()