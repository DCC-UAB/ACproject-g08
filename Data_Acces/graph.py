import pandas as pd
import csv
import networkx as nx
from itertools import combinations
import os
import pickle
import sys

def extreure_users():
    # Lee el CSV en un DataFrame
    df = pd.read_csv('./Data/ratings.csv')

    # Crea una nueva columna 'ratings'
    df['ratings'] = df.apply(lambda row: {'movieId': row['movieId'].astype(int), 'rating': row['rating']}, axis=1)

    # Agrupa por userId y crea la lista de diccionarios
    df_final = df.groupby('userId')['ratings'].apply(list).reset_index(name='ratings')

    # Guarda el DataFrame en un CSV
    df_final.to_csv('./Data/users.csv', index=False)

def extreure_movies():
    # Lee el CSV en un DataFrame
    df = pd.read_csv('./Data/ratings.csv')

    # Crea una nueva columna 'ratings'
    df['users'] = df.apply(lambda row: {'userId': row['userId'].astype(int), 'rating': row['rating']}, axis=1)

    # Agrupa por userId y crea la lista de diccionarios
    df_final = df.groupby('movieId')['users'].apply(list).reset_index(name='ratings')

    # Guarda el DataFrame en un CSV
    df_final.to_csv('./Data/movies.csv', index=False)

def conexions_usuaris():
    movies = pd.read_csv('./Data/movies.csv')

    with open('./Data/users.csv', 'r', encoding='utf-8') as infile, open('./Data/connections.csv', 'w', encoding='utf-8', newline='') as outfile:
        lector = csv.reader(infile)
        escritor = csv.writer(outfile)
        next(lector)

        # Escribe la cabecera del archivo de salida
        escritor.writerow(['userId', 'neighbours'])

        for fila in lector:
            veins = set()
            for dict in eval(fila[1]):
                for nou in eval(movies[movies['movieId'] == dict['movieId']]['ratings'].values[0]):
                    if str(fila[0]) != str(nou['userId']):
                        veins.add(nou['userId'])
            escritor.writerow([fila[0], veins])


def cargar_csv_usuarios(filepath):
    """Carga el archivo CSV de usuarios y devuelve un diccionario."""
    csv.field_size_limit(10_000_000)
    
    usuarios = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = int(row['userId'])
            ratings = eval(row['ratings'])  # Convierte la cadena de lista de diccionarios en una estructura Python
            usuarios[user_id] = ratings
    return usuarios

def construir_grafo(usuarios):
    """Construye un grafo no dirigido donde dos usuarios están conectados si han valorado la misma película."""
    G = nx.Graph()
    
    # Crear nodos para cada usuario
    G.add_nodes_from(usuarios.keys())

    # Diccionario para mapear películas a usuarios
    pelicula_a_usuarios = {}

    # Llenar el mapa película -> usuarios
    for user_id, ratings in usuarios.items():
        for rating in ratings:
            movie_id = rating['movieId']
            if movie_id not in pelicula_a_usuarios:
                pelicula_a_usuarios[movie_id] = []
            pelicula_a_usuarios[movie_id].append(user_id)

    # Agregar aristas entre usuarios que han valorado las mismas películas
    for users in pelicula_a_usuarios.values():
        if len(users) > 1:
            for user1, user2 in combinations(users, 2):
                G.add_edge(user1, user2)

    return G

def main():
    pickle_file = './Data/grafo.pickle'
    
    if os.path.exists(pickle_file):
        print("Cargando grafo desde archivo guardado...")
        with open(pickle_file, 'rb') as f:
            grafo = pickle.load(f)
    else:
        print("Cargando usuarios...")
        usuarios = cargar_csv_usuarios('./Data/users_small.csv')

        print("Construyendo grafo...")
        grafo = construir_grafo(usuarios)

        print("Guardando grafo en archivo...")
        with open(pickle_file, 'wb') as f:
            pickle.dump(grafo, f)
        
        nx.write_graphml(grafo, 'grafo.graphml')
        print("Grafo exportado a 'grafo.graphml'.")

    print("Grafo cargado o construido:")
    print(f"Número de nodos: {grafo.number_of_nodes()}")
    print(f"Número de aristas: {grafo.number_of_edges()}")

main()