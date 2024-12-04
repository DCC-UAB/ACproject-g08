import pandas as pd 
import organizers
import time

def metadata_extractor(movieId: str, df1):
    metadata = df1[df1['id'] == movieId]

    col_order = ['title', 'id', 'genres', 'popularity', 'release_date', 'vote_average', 'vote_count']
    col_names = ['Títol', 'ID', 'Gèneres', 'Popularitat', 'Data de llançament', 'Mitjana de vots', 'Número de vots']

    for col, nom in zip(col_order, col_names):
        val = metadata.iloc[0][col]
        print(f"{nom}: {val}") 

    return metadata

if __name__ == "__main__":
    df1 = pd.read_csv('./Data/movies_metadata.csv')
    df2 = pd.read_csv("./Data/ratings.csv")

    # start = time.time()

    # cuenta = organizers.ratings_organizer(organizers.rating_counter(df2))

    # lista = []
    # for ide in cuenta:
    #     lista.append(str(ide[0]))

    # metadata_searcher(lista, df1)

    # end = time.time()
    # print(f"\n{end - start}")

    # print(rating_counter('./Data/ratings_small.csv'))
