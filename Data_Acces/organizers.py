import pandas as pd
import time

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

if __name__ == "__main__":
    ratings = pd.read_csv("./Data/ratings.csv")
    start = time.time()
    cuenta = ratings_organizer(rating_counter(ratings))
    end = time.time()
    print(cuenta)
    print(end - start)