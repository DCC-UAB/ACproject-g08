import pandas as pd
import time
from extreure_metadata import *

# def ratings_per_pelicula(dataset):
#     #Dataset -> Direcció del dataset
#     #Return -> Dict{MovieId: ratings_count}
#     ratings = pd.read_csv(dataset)
#     cuenta = {}
#     for index, row in ratings.iterrows():
#         movie = int(row["movieId"])
#         # if movie not in cuenta:
#         #     cuenta[movie] = 0
#         # cuenta[movie] += 1
#         cuenta[movie] = cuenta.get(movie, 0) + 1
#     return cuenta

def rating_counter(df): # Usamos un dataset ya inicializado para normalizarlo a las posibles futuras verisones del código
    #df -> object Dataframe
    #Return -> dict{MovieId: ratings_count}
    cuenta = {}
    movies = list(df["movieId"]) # En vez de iterar filas y acceder a la columna iteramos sobre columna directamente
    for movie in movies:
        cuenta[movie] = cuenta.get(movie, 0) + 1 # En vez de un if usamos la funcion get de los objetos diccionario
    return cuenta

def rating_average(df): # Usamos un dataset ya inicializado para normalizarlo a las posibles futuras verisones del código
    #df -> object Dataframe
    #Return -> dict{MovieId: ratings_count}
    cuenta = {}
    ratings = {}
    movies = list(df["movieId"]) # En vez de iterar filas y acceder a la columna iteramos sobre columna directamente
    ratings_list = list(df["rating"])
    for index, movie in enumerate(movies):
        cuenta[movie] = cuenta.get(movie, 0) + 1
        ratings[movie] = ratings.get(movie, 0) + ratings_list[index]
    for movie in ratings:
        ratings[movie] = ratings[movie] / cuenta[movie]
    return ratings

def ratings_organizer(data: dict, returns: int = 5): # Funcion auxiliar para ordenar y devolver el número de peliculas deseadas
    organized = list(sorted(data.items(), key= lambda x: x[1], reverse= True))
    while returns < 0: # Control de errores de usuario
        if returns == -1:
            return organized
        cont = input("\nEl nombre de valors a retornar ha de ser un número major o igual a zero.\n\nDesitja introduïr un nou valor? y/n ")
        if cont == 'n' or cont == 'N':
            return None
        if cont == 'y' or cont == 'Y':
            try:
                returns = input("Escrigui el nou nombre de valors a retornar (predeterminat = 5): ")
                if returns == "":
                    returns = 5
                returns = int(returns)
            except:
                returns = -2
        else: 
            returns = -2
    return organized[:returns]

def metadata_organizer(movieIds: list, df1):
    """
    La funció rep una llista de IDs de pel·lícues i retorna un diccionari
    on la clau es la ID d'una pel·licula i el valor són les metadades de la pel·lícula.

    movieIds --> Llista de IDs de les pel·lícules que volem extreure les metadades
    df1 --> Pandas DataFrame on la funció realitza la cerca de les metadades
    """
    
    metadatas = {}
    if not isinstance(movieIds, list):
        print("Error: La entrada ha de ser una lista de IDs de pel·lícules.")
        return None

    for movieId in movieIds:
        try:
            metadata = metadata_extractor(movieId, df1)
            metadatas[movieId] = metadata
        except:
            if type(movieId) == int:
                print('Error: La ID de pel·lícula ha de ser format String (no Int)')   
            elif len(movieId) > 0:    
                print(f"Error: No s'ha pogut trobar metadata per la pel·lícula amb ID: {movieId}")
            else:
                print("Error: ID de pel·lícula nul no vàlid")
        print()

    return metadatas


if __name__ == "__main__":
    df1 = pd.read_csv('./Data/movies_metadata.csv')
    # metadata_organizer(['862', '21032', '', '31', '9598', 70], df1)
    ratings = pd.read_csv("./Data/ratings.csv")
    start = time.time()
    # cuenta = ratings_organizer(rating_counter(ratings))
    cuenta = ratings_organizer(rating_average(ratings))
    end = time.time()
    print(end - start)