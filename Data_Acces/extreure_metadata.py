import pandas as pd 
import organizers
import time

def metadata_extractor(movieIds: list, movies_metadata: pd.DataFrame):
    """
    Retorna les metadades de les pel·lícules segons les seves IDs.

    Arguments:
    movieIds -- Llista de IDs de les pel·lícules
    movies_metadata -- DataFrame amb les metadades de les pel·lícules
    n -- Nombre màxim de pel·lícules per a les quals volem extreure metadades
    """
    counter = 1
    for i,movieId in enumerate(movieIds):
        try:
            # Filtrar les metadades per la ID de la pel·lícula
            metadata = movies_metadata[movies_metadata['id'] == str(movieId)]

            if metadata.empty:
                print(f"\nPel·lícula amb ID {movieId} no es troba a la base de dades de metadades.")
                continue

            # Columnes i noms per mostrar
            col_order = ['title', 'id', 'genres', 'popularity', 'release_date', 'vote_average', 'vote_count']
            col_names = ['Títol', 'ID', 'Gèneres', 'Popularitat', 'Data de llançament', 'Mitjana de vots', 'Número de vots']

            print(f"\nMetadades per a la pel·lícula amb ID {movieId} (Recomanació {counter}) ({i+1}):")
            for col, nom in zip(col_order, col_names):
                val = metadata.iloc[0][col]
                print(f"{nom}: {val}")

            counter += 1

        except Exception as e:
            print(f"\nError processant la pel·lícula amb ID {movieId}: {e}")

def movie_finder(movieIds: list, movies_metadata: pd.DataFrame, n: int):
    ids = []
    counter = 0
    for movieId in movieIds:
        if counter >= n:
            break  # Hem arribat al límit de pel·lícules desitjades

        try:
            metadata = movies_metadata[movies_metadata['id'] == str(movieId)]

            if metadata.empty:
                print(f"\nPel·lícula amb ID {movieId} no es troba a la base de metadades.")
                continue

            ids.append(movieId) 
            counter += 1

        except Exception as e:
            print(f"\nNo es troben metadades per la pel·lícula amb ID {movieId}: {e}")
    return ids

if __name__ == "__main__":
    df1 = pd.read_csv('./Data/movies_metadata.csv')
    df2 = pd.read_csv('./Data/ratings.csv')

    # start = time.time()

    # cuenta = organizers.ratings_organizer(organizers.rating_counter(df2))

    # lista = []
    # for ide in cuenta:
    #     lista.append(str(ide[0]))

    # metadata_searcher(lista, df1)

    # end = time.time()
    # print(f"\n{end - start}")

    # print(rating_counter('./Data/ratings_small.csv'))
