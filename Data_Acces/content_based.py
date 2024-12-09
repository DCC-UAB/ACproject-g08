import pandas as pd
import numpy as np

def rate_usuari(df, user = 1):
    #Retorna els IDs de totes les películes que l'usuari ha donat rating
    assert user in df['userId'], "User not in database"
    ids = list(df1["userId"])
    ratings = {}
    movies = []
    for index, id in enumerate(ids):
        if id == user:
            movies.append(df1["movieId"][index])
            ratings[df1["movieId"][index]] = df1["rating"][index]
    return ratings

df1 = pd.read_csv('./Data/ratings_small.csv')
df2 = pd.read_csv('./Data/keywords.csv')

movies = rate_usuari(df1)

filter = df2[df2['id'].isin(movies)] #Agafa els IDs de les películes vistes per l'usuari que estiguin a l'arxiu keywords
rest = df2[~df2['id'].isin(movies)] #Agafa els IDs de les películes NO vistes per l'ususari que estiguin a l'arxiu keywords
print(movies)
print(filter)
print(rest.iloc[2297]) #Agafa la línia 2297, utilitzada per comprobar si el 2297 estaba aquí, no està perquè mostra el 2298