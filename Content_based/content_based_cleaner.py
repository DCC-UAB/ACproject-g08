import pandas as pd
import csv

# 1. Netejar els movie ids que no contenen cap keyword ni genere (ja que la seva similitud a altres pel·licules sempre serà 0)
def clean_empty_keywords():
    with open('./Data/keywords_genres.csv', 'r', encoding='utf-8') as infile, open('./Data/content_keywords.csv', 'w', encoding='utf-8', newline='') as outfile:
        lector = csv.reader(infile)
        escritor = csv.writer(outfile)

        # Escribe la cabecera del archivo de salida
        escritor.writerow(next(lector))

        for fila in lector:
            # Comprueba si la lista de palabras clave está vacía
            if len(eval(fila[1])) > 0 or len(eval(fila[2])):  # Elimina espacios en blanco de la lista de palabras clave
                escritor.writerow(fila)

# 2. Netejar pelicules sense keywords
def clean_ratings():
    ratings = pd.read_csv('./Data/ratings.csv')
    keywords = pd.read_csv('./Data/content_keywords.csv')
    ids = list(keywords['id'])
    
    # Filtrar el DataFrame
    ddf_filtrado = ratings[ratings['movieId'].isin(ids)]
    
    # Guardar el DataFrame filtrado en un nuevo CSV
    ddf_filtrado.to_csv('./Data/content_ratings.csv', index=False)

# 3. Netejar les movieId de keywords no presents a ratings
def clean_keywords():
    keywords = pd.read_csv('./Data/content_keywords.csv')
    ratings = pd.read_csv('./Data/content_ratings.csv')
    ids = list(keywords['id'])
    mids = list(ratings['movieId'])
    mids = set(mids)
    eliminadas = []
    for ide in ids:
        if ide not in mids:
            eliminadas.append(ide)
    keyword_filtrado = keywords[~keywords['id'].isin(eliminadas)]
    keyword_filtrado.to_csv('./Data/clean_content_keywords.csv', index=False)

def afegir_generes():
    keys = pd.read_csv('./Data/keywords.csv')
    meta = pd.read_csv('./Data/movies_metadata.csv')

    genres = meta[['id','genres']]
    genres = genres.drop([19730, 29503, 35587])
    genres = genres.reset_index(drop=True)

    keys['id'] = keys['id'].astype(int)
    genres['id'] = genres['id'].astype(int)

    keys = keys.merge(genres[['id', 'genres']], on='id', how='left')
    keys['genres'] = keys['genres'].apply(lambda x: [] if pd.isna(x)  else x)

    keys.to_csv('./Data/keywords_genres.csv', index=False)

if __name__ == '__main__':
    # afegir_generes()
    # clean_empty_keywords()
    # clean_ratings()
    # clean_keywords()
    pass