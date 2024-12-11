import pandas as pd 
import numpy as np
import csv
import ast

data = {}

df = pd.read_csv('./Data/keywords.csv')

lists = list(df["keywords"])
movies = list(df['id'])
total = 0
count = 0

for mid, string in zip(movies, lists):
    lista = ast.literal_eval(string)
    total += len(lista)
    count += 1
    for dict in lista:
        try:
            data[dict['id']]['count'] += 1
            data[dict['id']]['movies'].append(mid)
        except:
            data[dict['id']] = {'id' : dict['id'], 'name' : dict['name'], 'movies': [mid], 'count' : 1}

# Nombre del archivo CSV
filename = 'inverted_keywords.csv'

# Abrir el archivo para escritura
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    # Obtener los nombres de las columnas
    fieldnames = list(data[next(iter(data))].keys())
    
    # Crear un objeto DictWriter
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # Escribir los encabezados
    writer.writeheader()
    
    # Escribir las filas
    for key, value in data.items():
        writer.writerow(value)

print(f'El diccionario se ha guardado en {filename}')
print(f'El average de keywords por movie es {total/count}')
