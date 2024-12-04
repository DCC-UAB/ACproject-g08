import pandas as pd 
import numpy as np
import csv
import ast

data = {}

df = pd.read_csv('./Data/keywords.csv')

lists = list(df["keywords"])

for string in lists:
    lista = ast.literal_eval(string)
    for dict in lista:
        try:
            data[dict['id']]['count'] += 1
        except:
            data[dict['id']] = {'id' : dict['id'], 'name' : dict['name'], 'count' : 1}

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
