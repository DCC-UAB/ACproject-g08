import pandas as pd
import numpy as np
import time
from extreure_metadata import *

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
            d = d * ratings_user[vista['id']]**2 #Fem que el rating a la película vista importi
            dists.append(d)
        distancies[no_vista['id']] = max(dists)

    distancies_sort = dict(sorted(distancies.items(), key=lambda item: item[1], reverse = True))
    p_item = list(distancies_sort.items())
    p_key = list(distancies_sort.keys())
    
    a = list(item[1] for item in p_item)
    mini = min(a)
    maxi = max(a)
    normalitzat = []
    final =  []
    for peli, puntuacio in p_item:
        # Comentaris utilitzats per a la normalització
        # normalitzat = (puntuacio - mini) / (maxi - mini) * 5
        # normalitzat.append([peli, normalitzat])
        final.append(peli)
    metadata = pd.read_csv('./Data/movies_metadata.csv')
    f2 = movie_finder(final, metadata, num)


    # recommendations = []
    # for i in range(num):
    #     recommendations.append(p_key[i])
    # return recommendations

    return f2

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
    
    a = list(item[1] for item in p_item)
    mini = min(a)
    maxi = max(a)
    normalitzat = []
    final =  []
    for peli, puntuacio in p_item:
        # Comentaris utilitzats per a la normalització
        # normalitzat = (puntuacio - mini) / (maxi - mini) * 5
        # normalitzat.append([peli, normalitzat])
        final.append(peli)
    metadata = pd.read_csv('./Data/movies_metadata.csv')
    f2 = movie_finder(final, metadata, num)


    # recommendations = []
    # for i in range(num):
    #     recommendations.append(p_key[i])
    # return recommendations

    return f2

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
            d = (dk + (dg * 1.5)) * ratings_user[vista['id']]**2 #Fem que el rating a la película vista importi
            dists.append(d)
        distancies[no_vista['id']] = max(dists)

    distancies_sort = dict(sorted(distancies.items(), key=lambda item: item[1], reverse = True))
    p_item = list(distancies_sort.items())
    p_key = list(distancies_sort.keys())
    
    a = list(item[1] for item in p_item)
    mini = min(a)
    maxi = max(a)
    normalitzat = []
    final =  []
    for peli, puntuacio in p_item:
        # Comentaris utilitzats per a la normalització
        # normalitzat = (puntuacio - mini) / (maxi - mini) * 5
        # normalitzat.append([peli, normalitzat])
        final.append(peli)
    metadata = pd.read_csv('./Data/movies_metadata.csv')
    f2 = movie_finder(final, metadata, num)


    # recommendations = []
    # for i in range(num):
    #     recommendations.append(p_key[i])
    # return recommendations

    return f2

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
            d = (dk + (dg * 1.5))
            dists.append(d)
        distancies[no_vista['id']] = max(dists)

    distancies_sort = dict(sorted(distancies.items(), key=lambda item: item[1], reverse = True))
    #print(distancies_sort)
    p_item = list(distancies_sort.items())
    p_key = list(distancies_sort.keys())
    
    a = list(item[1] for item in p_item)
    mini = min(a)
    maxi = max(a)
    normalitzat = []
    final =  []
    for peli, puntuacio in p_item:
        # Comentaris utilitzats per a la normalització
        # normalitzat = (puntuacio - mini) / (maxi - mini) * 5
        # normalitzat.append([peli, normalitzat])
        final.append(peli)
    
    metadata = pd.read_csv('./Data/movies_metadata.csv')
    f2 = movie_finder(final, metadata, num)


    # recommendations = []
    # for i in range(num):
    #     recommendations.append(p_key[i])
    # return recommendations

    return f2

def main_content():
    # u = 0
    # n = 10
    u = int(input("Introdueix ID d'usuari per recomanar-li pel·lícules: "))
    n = int(input("Introdueix el número de pel·lícules a recomanar: "))

    print("1. Recomanador sense ratings ni gèneres")
    print("2. Recomanador amb ratings però sense gèneres")
    print("3. Recomanador sense ratings però amb gèneres")
    print("4. Recomanador amb ratings i gèneres")
    print()
    
    opcio = int(input("Escolleix el tipus de recomanador: "))

    df1 = pd.read_csv('./Data/content_ratings.csv')
    df2 = pd.read_csv('./Data/content_keywords.csv')
    metadata = pd.read_csv('./Data/movies_metadata.csv')
    if opcio == 1:
        final = content_recommend(df1, df2, u, n)
        metadata_extractor(final, metadata)
    # user 1 -> [44284, 52856, 11017, 218473, 23637]
    # user 2 -> [604, 52587, 24100, 251797, 433878]
    elif opcio == 2:
        final = content_recommend_basic(df1, df2, u, n)
        metadata_extractor(final, metadata)
    # user 1 -> [44284, 52856, 11017, 218473, 23637]
    # user 2 -> [604, 251797, 433878, 52587, 31586]
    elif opcio == 3:
        final = content_recommend_genres(df1, df2, u, n)
        metadata_extractor(final, metadata)
    # user 1 -> [44284, 213917, 124676, 41240, 124843]
    # user 2 -> [604, 14886, 9684, 31208, 308084]
    elif opcio == 4:
        final = content_recommend_genres_basic(df1, df2, u, n)
        metadata_extractor(final, metadata)
    # user 1 -> [44284, 213917, 124676, 41240, 124843]
    # user 2 -> [604, 285, 22, 17745, 1865]
    #print(final)

if __name__ == "__main__":
    main_content()
    # Experiment usuari 0 -> Igual que l'usuari 2 però no té la 605 i 628
    # [251797, 433878, 10131, 10072, 33494]
    # [251797, 433878, 31586, 81367, 29968]
    # [14886, 9684, 31208, 308084, 8049]
    # [285, 22, 17745, 1865, 48764]

    # Usuari 55 (recomendaciones) -> 47 ratings | Exemple amb puntuacions per veririficar valors diferents -> retorna (ID, puntuacio)
    # [4729, 607, 15247, 251797, 433878, 3146, 34388, 46886, 4893, 31586]
    # [4729, 607, 15247, 46886, 251797, 4893, 433878, 3146, 34388, 31586]
    # [607, 15247, 34388, 4729, 11595, 13257, 27440, 26147, 29959, 43015]
    # [4729, 607, 15247, 34388, 11595, 11915, 13257, 27440, 220029, 26147]

    # [(4729, 1.7320508075688772), (607, 1.7200522903844537), (15247, 1.4907119849998598), (251797, 1.4907119849998598), (433878, 1.4907119849998598), (3146, 1.3153341044116411), (34388, 1.1768778828946262), (46886, 1.1547005383792515), (4893, 1.1547005383792515), (31586, 1.118033988749895)]
    # [(4729, 1.0), (607, 0.7692307692307693), (15247, 0.6666666666666666), (46886, 0.6666666666666666), (251797, 0.6666666666666666), (4893, 0.6666666666666666), (433878, 0.6666666666666666), (3146, 0.5882352941176471), (34388, 0.5263157894736842), (31586, 0.5)]
    # [(607, 5.074154256634138), (15247, 4.844813951249544), (34388, 4.530979849144311), (4729, 4.330127018922193), (11595, 4.2485291572496005), (13257, 4.167217594431427), (27440, 4.167217594431427), (26147, 4.167217594431427), (29959, 4.1433024288966696), (43015, 4.1433024288966696)]
    # [(4729, 2.5), (607, 2.269230769230769), (15247, 2.1666666666666665), (34388, 2.026315789473684), (11595, 1.9), (11915, 1.9), (13257, 1.8636363636363638), (27440, 1.8636363636363638), (220029, 1.8636363636363638), (26147, 1.8636363636363638)]

    #Experiment usuari 0.2 -> Igual que l'usuari 55 menys 1480 i 1649
    # [(4729, 607, 15247, 251797, 433878, 3146, 34388, 46886, 14893, 31586]
    # [(4729, 607, 15247, 46886, 251797, 4893, 433878, 3146, 34388, 31586]
    # [(607, 15247, 34388, 4729, 11595, 13257, 27440, 26147, 29959, 43015]
    # [(4729, 607, 15247, 34388, 11595, 11915, 13257, 27440, 220029, 26147]

    # [(4729, 1.7320508075688772), (607, 1.7200522903844537), (15247, 1.4907119849998598), (251797, 1.4907119849998598), (433878, 1.4907119849998598), (3146, 1.3153341044116411), (34388, 1.1768778828946262), (46886, 1.1547005383792515), (4893, 1.1547005383792515), (31586, 1.118033988749895)]
    # [(4729, 1.0), (607, 0.7692307692307693), (15247, 0.6666666666666666), (46886, 0.6666666666666666), (251797, 0.6666666666666666), (4893, 0.6666666666666666), (433878, 0.6666666666666666), (3146, 0.5882352941176471), (34388, 0.5263157894736842), (31586, 0.5)]
    # [(607, 5.074154256634138), (15247, 4.844813951249544), (34388, 4.530979849144311), (4729, 4.330127018922193), (11595, 4.2485291572496005), (13257, 4.167217594431427), (27440, 4.167217594431427), (26147, 4.167217594431427), (29959, 4.1433024288966696), (43015, 4.1433024288966696)]
    # [(4729, 2.5), (607, 2.269230769230769), (15247, 2.1666666666666665), (34388, 2.026315789473684), (11595, 1.9), (11915, 1.9), (13257, 1.8636363636363638), (27440, 1.8636363636363638), (220029, 1.8636363636363638), (26147, 1.8636363636363638)]

    #Experiment usuari 0.2 amb ratings**2
    # [607, 15247, 251797, 433878, 3146, 34388, 31586, 81367, 29968, 57240]
    # [4729, 607, 15247, 46886, 251797, 4893, 433878, 3146, 34388, 31586]
    # [607, 15247, 34388, 11595, 13257, 27440, 26147, 29959, 43015, 965]
    # [4729, 607, 15247, 34388, 11595, 11915, 13257, 27440, 220029, 26147]

    #Normalitzat per a valors entre 0 i 5:
    # [[607, 5.0], [15247, 4.333333333333332], [251797, 4.333333333333332], [433878, 4.333333333333332], [3146, 3.8235294117647056], [34388, 3.421052631578947], [31586, 3.2499999999999996], [81367, 3.2499999999999996], [29968, 3.2499999999999996], [57240, 3.2499999999999996]]
    # [[4729, 5.0], [607, 3.8461538461538463], [15247, 3.333333333333333], [46886, 3.333333333333333], [251797, 3.333333333333333], [4893, 3.333333333333333], [433878, 3.333333333333333], [3146, 2.9411764705882355], [34388, 2.631578947368421], [31586, 2.5]]
    # [[607, 5.0], [15247, 4.7740112994350286], [34388, 4.464763603925067], [11595, 4.186440677966102], [13257, 4.1063174114021574], [27440, 4.1063174114021574], [26147, 4.1063174114021574], [29959, 4.082751744765703], [43015, 4.082751744765703], [965, 4.03954802259887]]
    # [[4729, 5.0], [607, 4.538461538461538], [15247, 4.333333333333333], [34388, 4.052631578947368], [11595, 3.8], [11915, 3.8], [13257, 3.7272727272727275], [27440, 3.7272727272727275], [220029, 3.7272727272727275], [26147, 3.7272727272727275]]

    # Metadades per a la pel·lícula amb ID 607 (Recomanació 1):
    # Títol: Men in Black
    # ID: 607
    # Gèneres: [{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}, {'id': 35, 'name': 'Comedy'}, {'id': 878, 'name': 'Science Fiction'}]

    # Metadades per a la pel·lícula amb ID 15247 (Recomanació 2):
    # Títol: The Ipcress File
    # ID: 15247
    # Gèneres: [{'id': 53, 'name': 'Thriller'}]

    # Metadades per a la pel·lícula amb ID 251797 (Recomanació 3):
    # Títol: The Unknown Woman
    # ID: 251797
    # Gèneres: [{'id': 99, 'name': 'Documentary'}]

    # Metadades per a la pel·lícula amb ID 433878 (Recomanació 4):
    # Títol: The Mars Generation
    # ID: 433878
    # Gèneres: [{'id': 99, 'name': 'Documentary'}]

    # Metadades per a la pel·lícula amb ID 3146 (Recomanació 5):
    # Títol: The War of the Gargantuas
    # ID: 3146
    # Gèneres: [{'id': 28, 'name': 'Action'}, {'id': 27, 'name': 'Horror'}, {'id': 878, 'name': 'Science Fiction'}]

    # Metadades per a la pel·lícula amb ID 34388 (Recomanació 6):
    # Títol: Funeral in Berlin
    # ID: 34388
    # Gèneres: [{'id': 53, 'name': 'Thriller'}]

    # Metadades per a la pel·lícula amb ID 31586 (Recomanació 7):
    # Títol: North
    # ID: 31586
    # Gèneres: [{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}, {'id': 10751, 'name': 'Family'}, {'id': 14, 'name': 'Fantasy'}, {'id': 878, 'name': 'Science Fiction'}]

    # Metadades per a la pel·lícula amb ID 81367 (Recomanació 8):
    # Títol: Slappy and the Stinkers
    # ID: 81367
    # Gèneres: [{'id': 12, 'name': 'Adventure'}, {'id': 10751, 'name': 'Family'}]

    # Metadades per a la pel·lícula amb ID 29968 (Recomanació 9):
    # Títol: Nothing in Common
    # ID: 29968
    # Gèneres: [{'id': 18, 'name': 'Drama'}, {'id': 35, 'name': 'Comedy'}, {'id': 10749, 'name': 'Romance'}]

    # Metadades per a la pel·lícula amb ID 57240 (Recomanació 10):
    # Títol: Big Shots
    # ID: 57240
    # Gèneres: [{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}, {'id': 35, 'name': 'Comedy'}]

    # Manualment podem comprobar que les recomanacions són correctes