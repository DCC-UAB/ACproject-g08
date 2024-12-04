import pandas as pd 
import organizers
import time

def metadata_searcher(movieIds: list, df1):
    metadatas = {}
    for movieId in movieIds:
        try:
            metadata = df1[df1['id'] == movieId]
            metadatas[movieId] = metadata

            col_order = ['title', 'id', 'genres', 'popularity', 'release_date', 'vote_average', 'vote_count']
            col_names = ['Títol', 'ID', 'Gèneres', 'Popularitat', 'Data de llançament', 'Mitjana de vots', 'Número de vots']

            for col, nom in zip(col_order, col_names):
                val = metadata.iloc[0][col]
                print(f"{nom}: {val}") 
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
    df2 = pd.read_csv("./Data/ratings.csv")

    start = time.time()

    cuenta = organizers.ratings_organizer(organizers.rating_counter(df2))

    lista = []
    for ide in cuenta:
        lista.append(str(ide[0]))

    metadata_searcher(lista, df1)

    end = time.time()
    print(f"\n{end - start}")

    # metadata_searcher(['862', '21032', '', '31', '9598', 70], df1)

    # print(rating_counter('./Data/ratings_small.csv'))
