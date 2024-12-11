import pandas as pd

def matrix_convertion():
    df = pd.read_csv('./Data/ratings.csv')

    df['userId'] = df['userId'].astype(int)
    df['movieId'] = df['movieId'].astype(int)
    df['rating'] = df['rating'].astype(float)
    
    df = df.dropna(subset=['userId', 'movieId', 'rating'])

    user_movie_matrix = df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )      

    user_movie_matrix.to_csv('./Data/user_movie_matrix.csv', index=True)

matrix_convertion()