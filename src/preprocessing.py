import pandas as pd
import numpy as np
import os
import re

def get_data_paths():
    """Get absolute paths to data files."""
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_path)
    data_dir = os.path.join(project_root, 'data')
    return data_dir, os.path.join(data_dir, 'ratings.csv'), os.path.join(data_dir, 'movies.csv')

def load_data():
    """Load ratings and movies datasets with robust error handling."""
    data_dir, ratings_path, movies_path = get_data_paths()
    
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        raise FileNotFoundError(f"CSV files not found in {data_dir}")
        
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    # Clean columns
    ratings.columns = ratings.columns.str.strip()
    movies.columns = movies.columns.str.strip()
    
    return ratings, movies

def clean_data(ratings, movies, min_user_ratings=5, min_movie_ratings=5):
    """Clean data by removing missing values and filtering items."""
    ratings = ratings.dropna().drop_duplicates()
    movies = movies.dropna().drop_duplicates()

    # Dynamic column names
    user_col = 'userId' if 'userId' in ratings.columns else 'userID'
    movie_col = 'movieId' if 'movieId' in ratings.columns else 'movieID'

    user_counts = ratings[user_col].value_counts()
    ratings = ratings[ratings[user_col].isin(user_counts[user_counts >= min_user_ratings].index)]

    movie_counts = ratings[movie_col].value_counts()
    ratings = ratings[ratings[movie_col].isin(movie_counts[movie_counts >= min_movie_ratings].index)]
    
    movies = movies[movies[movie_col].isin(ratings[movie_col].unique())]
    
    return ratings, movies

def preprocess():
    """Main preprocessing entry point."""
    print("Preprocessing data...")
    try:
        ratings, movies = load_data()
        ratings, movies = clean_data(ratings, movies)
        print(f"Data cleaned: {len(movies)} movies, {len(ratings)} ratings.")
        return ratings, movies
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return None, None

if __name__ == "__main__":
    preprocess()
