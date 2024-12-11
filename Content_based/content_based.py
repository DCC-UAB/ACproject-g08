import pandas as pd
import numpy as np
import time 

def rate_usuari(df, user = 1):
    #Retorna els IDs de totes les películes que l'usuari ha donat rating
    assert user in df['userId'], "User not in database"
    ids = list(df["userId"])
    ratings = {}
    for index, id in enumerate(ids):
        if id == user:
            ratings[df["movieId"][index]] = df["rating"][index]
    return ratings

def dice_coefficient(n1, n2):
    #Reb dues llistes de keywords i retorna el dice coefficient
    inter = 0 #Intersecció
    for keyword in n1:
        if keyword in n2:
            inter += 1
    dist = (2 * inter) / (len(n1) + len(n2))
    return dist

def keyword_list(movie, df):
    #Reb ID de movie i retorna la llista de keywords
    a = df[df['id'] == movie]["keywords"] #Extreure el string de la columna keywords
    b = eval(a[0]) #Passar string al format adeqüat (llista de diccionaris)
    keywords = list(x['name'] for x in b)
    return keywords

def content_recommend(ratings, keywords, user, num):
    # ratings -> df amb els ratings
    # keywords -> df amb les keywords
    # user -> usuari al que fer la recomanació
    # num -> número de recomanacions
    # Retorna els IDs de les num películes més recomanades
    ratings_user = rate_usuari(ratings, user)
    movies = list(ratings_user.keys())
    movies_valides = keywords[keywords['id'].isin(movies)] #Ens quedem amb les películes vistes que estiguin a l'arxiu keywords
    rest = keywords[~keywords['id'].isin(movies)]
    distancies = {}

    for index, no_vista in rest.iterrows():
        # print()
        # print(no_vista['id'])
        filtro = eval(no_vista['keywords'])
        keywords_no_vista = list(x['name'] for x in filtro)
        # print(keywords_no_vista)
        dists = []
        for index, vista in movies_valides.iterrows(): #Fem la distància de la película no vista amb cada película vista
            # print()
            # print(x['id'], ":", ratings[x['id']])
            filtro2 = eval(vista['keywords'])
            keywords_vista = list(x['name'] for x in filtro2)
            # print(keywords)
            d = dice_coefficient(keywords_vista, keywords_no_vista)
            d = d * ratings_user[vista['id']]**(1/2) #Fem que el rating a la película vista importi pero no tant
            dists.append(d)
        distancies[no_vista['id']] = max(dists)

    distancies_sort = dict(sorted(distancies.items(), key=lambda item: item[1], reverse = True))
    p_item = list(distancies_sort.items())
    p_key = list(distancies_sort.keys())

    recommendations = []
    for i in range(num):
        recommendations.append(p_key[i])
    return recommendations

df1 = pd.read_csv('./Data/ratings.csv')
df2 = pd.read_csv('./Data/content_keywords.csv')
final = content_recommend(df1, df2, 1, 5)
print(final)
[44284, 52856, 11017, 218473, 23637]