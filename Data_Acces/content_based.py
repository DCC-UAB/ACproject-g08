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

def dice_coefficient(n1, n2):
    #Reb dues llistes de keywords i retorna el dice coefficient
    inter = 0 #Intersecció
    for keyword in n2:
        if keyword in n2:
            inter += 1
    dist = inter / (len(n1) + len(n2))
    return dist

def keyword_list(movie, file):
    #Reb ID de movie i retorna la llista de keywords
    df = pd.read_csv(file)
    a = df[df['id'] == movie]["keywords"] #Extreure el string de la columna keywords
    b = eval(a[0]) #Passar string al format adeqüat (llista de diccionaris)
    keywords = list(x['name'] for x in b)
    return keywords

df1 = pd.read_csv('./Data/ratings_small.csv')
df2 = pd.read_csv('./Data/keywords.csv')

ratings = rate_usuari(df1)
movies = list(ratings.keys())

filter = df2[df2['id'].isin(movies)] #Agafa els IDs de les películes vistes per l'usuari que estiguin a l'arxiu keywords
rest = df2[~df2['id'].isin(movies)] #Agafa els IDs de les películes NO vistes per l'ususari que estiguin a l'arxiu keywords
print(movies)
print(filter)
print(rest.iloc[2297]) #Agafa la línia 2297, utilitzada per comprobar si el 2297 estaba aquí, no està perquè mostra el 2298