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
    if len(n1) == 0 or len(n2) == 0:
        return 0
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

def content_recommend_basic(ratings, keywords, user, num):
    # ratings -> df amb els ratings
    # keywords -> df amb les keywords
    # user -> usuari al que fer la recomanació
    # num -> número de recomanacions
    # Retorna els IDs de les num películes més recomanades en base a NOMÉS la similitud a películes ja vistes per l'usuari
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
            dists.append(d)
        distancies[no_vista['id']] = max(dists)

    distancies_sort = dict(sorted(distancies.items(), key=lambda item: item[1], reverse = True))
    p_item = list(distancies_sort.items())
    p_key = list(distancies_sort.keys())

    recommendations = []
    for i in range(num):
        recommendations.append(p_key[i])
    return recommendations

def content_recommend_genres(ratings, keywords, user, num):
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
        filtro_g = eval(no_vista['genres'])
        genres_no_vista = list(x['name'] for x in filtro_g)
        # print(keywords_no_vista)
        dists = []
        for index, vista in movies_valides.iterrows(): #Fem la distància de la película no vista amb cada película vista
            # print()
            # print(x['id'], ":", ratings[x['id']])
            filtro2 = eval(vista['keywords'])
            keywords_vista = list(x['name'] for x in filtro2)
            filtro2_g = eval(vista['genres'])
            genres_vista = list(x['name'] for x in filtro2_g)
            # print(keywords)
            dk = dice_coefficient(keywords_vista, keywords_no_vista)
            dg = dice_coefficient(genres_vista, genres_no_vista)
            d = (dk + (dg * 1.5)) * ratings_user[vista['id']]**(1/2) #Fem que el rating a la película vista importi pero no tant
            dists.append(d)
        distancies[no_vista['id']] = max(dists)

    distancies_sort = dict(sorted(distancies.items(), key=lambda item: item[1], reverse = True))
    p_item = list(distancies_sort.items())
    p_key = list(distancies_sort.keys())

    recommendations = []
    for i in range(num):
        recommendations.append(p_key[i])
    return recommendations

def content_recommend_genres_basic(ratings, keywords, user, num):
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
        filtro_g = eval(no_vista['genres'])
        genres_no_vista = list(x['name'] for x in filtro_g)
        # print(keywords_no_vista)
        dists = []
        for index, vista in movies_valides.iterrows(): #Fem la distància de la película no vista amb cada película vista
            # print()
            # print(x['id'], ":", ratings[x['id']])
            filtro2 = eval(vista['keywords'])
            keywords_vista = list(x['name'] for x in filtro2)
            filtro2_g = eval(vista['genres'])
            genres_vista = list(x['name'] for x in filtro2_g)
            # print(keywords)
            dk = dice_coefficient(keywords_vista, keywords_no_vista)
            dg = dice_coefficient(genres_vista, genres_no_vista)
            d = (dk + (dg * 1.5)) #Fem que el rating a la película vista importi pero no tant
            dists.append(d)
        distancies[no_vista['id']] = max(dists)

    distancies_sort = dict(sorted(distancies.items(), key=lambda item: item[1], reverse = True))
    #print(distancies_sort)
    p_item = list(distancies_sort.items())
    p_key = list(distancies_sort.keys())

    recommendations = []
    for i in range(num):
        recommendations.append(p_key[i])
    return recommendations

df1 = pd.read_csv('./Data/content_ratings.csv')
df2 = pd.read_csv('./Data/content_keywords.csv')
final = content_recommend(df1, df2, 0, 5)
print(final)
# user 1 -> [44284, 52856, 11017, 218473, 23637]
# user 2 -> [604, 52587, 24100, 251797, 433878]
final = content_recommend_basic(df1, df2, 0, 5)
print(final)
# user 1 -> [44284, 52856, 11017, 218473, 23637]
# user 2 -> [604, 251797, 433878, 52587, 31586]
final = content_recommend_genres(df1, df2, 0, 5)
print(final)
# user 1 -> [44284, 213917, 124676, 41240, 124843]
# user 2 -> [604, 14886, 9684, 31208, 308084]
final = content_recommend_genres_basic(df1, df2, 0, 5)
# user 1 -> [44284, 213917, 124676, 41240, 124843]
# user 2 -> [604, 285, 22, 17745, 1865]
print(final)

# Experiment usuari 0 -> Igual que l'usuari 2 però no té la 605 i 628
# [251797, 433878, 10131, 10072, 33494]
# [251797, 433878, 31586, 81367, 29968]
# [14886, 9684, 31208, 308084, 8049]
# [285, 22, 17745, 1865, 48764]