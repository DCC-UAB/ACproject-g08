import pandas as pd

def ratings_per_pelicula(dataset):
    #Dataset -> Direcció del dataset
    #Return -> Dict{MovieId: ratings_count}
    ratings = pd.read_csv(dataset)
    cuenta = {}
    for index, row in ratings.iterrows():
        movie = int(row["movieId"])
        if movie not in cuenta:
            cuenta[movie] = 0
        cuenta[movie] += 1
    return cuenta

# cuenta = ratings_per_pelicula("Data/ratings_small.csv")
# print(cuenta)