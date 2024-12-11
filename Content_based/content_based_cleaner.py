import pandas as pd
import csv

# 1. Netejar els movie ids que no contenen cap keyword (ja que la seva similitud a altres pel·licules sempre serà 0)
def clean_empty_keywords():
    with open('./Data/keywords.csv', 'r', encoding='utf-8') as infile, open('./Data/content_keywords.csv', 'w', encoding='utf-8', newline='') as outfile:
        lector = csv.reader(infile)
        escritor = csv.writer(outfile)

        # Escribe la cabecera del archivo de salida
        escritor.writerow(next(lector))

        for fila in lector:
            # Comprueba si la lista de palabras clave está vacía
            if len(eval(fila[1])) > 0:  # Elimina espacios en blanco de la lista de palabras clave
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
if __name__ == '__main__':
    # clean_empty_keywords()
    # clean_ratings()
    pass