import pandas as pd
import numpy as np

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

def item_cleaner():
    # Cargar el archivo CSV
    df = pd.read_csv('./Data/ratings.csv')

    # Paso 1: Identificar los usuarios únicos (que tienen solo una valoración)
    usuarios_unicos = df['userId'].value_counts()
    usuarios_unicos = usuarios_unicos[usuarios_unicos == 1].index

    # Paso 2: Filtrar las filas que pertenecen a usuarios únicos
    df_usuarios_unicos = df[df['userId'].isin(usuarios_unicos)]

    # Paso 3: Contar las valoraciones por película realizadas solo por usuarios únicos
    conteo_unicos = df_usuarios_unicos['movieId'].value_counts()

    # Paso 4: Contar las valoraciones totales por película
    conteo_totales = df['movieId'].value_counts()

    # Paso 5: Identificar películas donde todas las valoraciones provienen de usuarios únicos
    conteo_unicos_alineado = conteo_unicos.reindex(conteo_totales.index, fill_value=0)
    peliculas_a_eliminar = conteo_unicos[conteo_unicos_alineado == conteo_totales].index
    print(peliculas_a_eliminar)

    # Paso 6: Eliminar las películas identificadas
    df_filtrado = df[~df['movieId'].isin(peliculas_a_eliminar)]

    # Guardar el resultado en un nuevo archivo CSV
    df_filtrado.to_csv('./Data/item_to_item.csv', index=False)

    print("Películas eliminadas y nuevo archivo guardado como 'archivo_filtrado.csv'.")

# def adri():
#     item1 = [0,1,5,3,1,4,1,3,0]
#     item2 = [2,5,1,4,1,1,4,1,2]
#     medias = [3.5,2.5,3.6,3.75,2.2,2.8,2.2,2,2.5]

#     numerador = 0
#     denominador1 = 0
#     denominador2 = 0
#     for i in range(len(item1)):
#         if item1[i] != 0 and item2[i] != 0:
#             numerador += (item1[i] - medias[i]) * (item2[i] - medias[i])
#             denominador1 += (item1[i] - medias[i])**2
#             denominador2 += (item2[i] - medias[i])**2
#         else:
#             if item1[i] != 0:
#                 denominador1 += (item1[i] - medias[i])**2
#             if item2[i] != 0:
#                 denominador2 += (item2[i] - medias[i])**2

        
#     denominador = np.sqrt(denominador1) * np.sqrt(denominador2)
#     print(numerador/denominador)

if __name__ == '__main__':
    item_cleaner()