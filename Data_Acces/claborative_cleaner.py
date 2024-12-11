import pandas as pd

def matrix_convertion():
    df = pd.read_csv('./Data/ratings.csv')
    user_movie_matrix = df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    )


matrix_convertion()