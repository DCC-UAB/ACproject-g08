import pandas as pd

def extreure_users():
    # Lee el CSV en un DataFrame
    df = pd.read_csv('./Data/ratings.csv')

    # Crea una nueva columna 'ratings'
    df['ratings'] = df.apply(lambda row: {'movieId': row['movieId'].astype(int), 'rating': row['rating']}, axis=1)

    # Agrupa por userId y crea la lista de diccionarios
    df_final = df.groupby('userId')['ratings'].apply(list).reset_index(name='ratings')

    # Guarda el DataFrame en un CSV
    df_final.to_csv('./Data/users.csv', index=False)

def extreure_movies():
    # Lee el CSV en un DataFrame
    df = pd.read_csv('./Data/ratings_small.csv')

    # Crea una nueva columna 'ratings'
    df['users'] = df.apply(lambda row: {'userId': row['userId'].astype(int), 'rating': row['rating']}, axis=1)

    # Agrupa por userId y crea la lista de diccionarios
    df_final = df.groupby('movieId')['users'].apply(list).reset_index(name='ratings')

    # Guarda el DataFrame en un CSV
    df_final.to_csv('./Data/movies_small.csv', index=False)

extreure_movies()
