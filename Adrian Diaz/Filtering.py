import pandas as pd 
import numpy as np

df1 = pd.read_csv('../data/credits.csv')
df2 = pd.read_csv('../data/movies_metadata.csv')

df1.columns = ['id','title','cast','crew']
df2 = df2.merge(df1, on='id')