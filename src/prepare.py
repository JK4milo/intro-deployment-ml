import pandas as pd


fin_data = pd.read_csv('dataset/finantials.csv')
movie_data = pd.read_csv('dataset/movies.csv')
opening_data = pd.read_csv('dataset/opening_gross.csv')

numeric_columns_mask = (movie_data.dtypes == float) | (movie_data.dtypes == int)
numeric_columns = [column for column in numeric_columns_mask.index if numeric_columns_mask[column]]
movie_data = movie_data[numeric_columns+['movie_title']]

fin_data = fin_data[['movie_title', 'production_budget', 'worldwide_gross']]

fin_movie_data = pd.merge(fin_data, movie_data, on='movie_title', how='left')
full_movie_data = pd.merge(opening_data, fin_movie_data, on='movie_title', how='left')

full_movie_data = full_movie_data.drop(['gross','movie_title'], axis=1)

full_movie_data.to_csv('dataset/full_data.csv',index=False)

