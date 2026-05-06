import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import pickle

def load_assets():
    """
    Load data and prepare the recommendation model.
    Uses st.cache_resource for fast subsequent loads in Streamlit.
    """
    # Robust path handling: Find the root directory
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_path)
    data_dir = os.path.join(project_root, 'data')
    
    movies_path = os.path.join(data_dir, 'movies.csv')
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        print(f"Error: Data files not found in {data_dir}")
        return None
        
    try:
        # Load datasets
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        
        # Clean column names (remove possible BOM or whitespace)
        movies.columns = movies.columns.str.strip()
        ratings.columns = ratings.columns.str.strip()
        
        # Handle cases where columns might be slightly different
        movie_col = 'movieId' if 'movieId' in movies.columns else 'movieID'
        user_col = 'userId' if 'userId' in ratings.columns else 'userID'
        
        # Ensure correct types
        movies[movie_col] = pd.to_numeric(movies[movie_col], errors='coerce')
        ratings[user_col] = pd.to_numeric(ratings[user_col], errors='coerce')
        ratings[movie_col] = pd.to_numeric(ratings[movie_col], errors='coerce')
        ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
        
        movies = movies.dropna(subset=[movie_col])
        ratings = ratings.dropna(subset=[user_col, movie_col, 'rating'])
        
        # Preprocess movies: extract year safely
        movies['year'] = movies['title'].astype(str).str.extract(r'\((\d{4})\)').fillna(1990).astype(int)
        
        # Create pivot table (using pivot_table to handle possible duplicates safely)
        movie_user_matrix = ratings.pivot_table(index=movie_col, columns=user_col, values='rating').fillna(0)
        
        # Fit NearestNeighbors model
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
        model_knn.fit(movie_user_matrix)
        
        return {
            'movies': movies,
            'ratings': ratings,
            'matrix': movie_user_matrix,
            'model': model_knn,
            'movie_col': movie_col,
            'user_col': user_col
        }
    except Exception as e:
        print(f"Error loading assets: {e}")
        return None

# Wrap load_assets with streamlit cache if available
try:
    load_assets = st.cache_resource(load_assets)
except Exception:
    # Fallback for non-streamlit environments
    pass

def get_recommendations(user_id, top_n=10, assets=None):
    """
    Generate movie recommendations for a given user.
    """
    if assets is None:
        assets = load_assets()
        
    if assets is None:
        return "⚠️ Data files not found or corrupted. Please check the 'data' folder."
        
    movies = assets['movies']
    ratings = assets['ratings']
    matrix = assets['matrix']
    model = assets['model']
    movie_col = assets['movie_col']
    user_col = assets['user_col']
    
    # 1. Find movies the user has already rated
    user_ratings = ratings[ratings[user_col] == user_id]
    
    if user_ratings.empty:
        # Fallback: Return top popular movies if user ID is unknown
        popular_movie_ids = ratings.groupby(movie_col).size().sort_values(ascending=False).head(top_n).index
        recs = movies[movies[movie_col].isin(popular_movie_ids)].copy()
        recs['predicted_rating'] = 5.0
        return recs[['title', 'genres', 'year', 'predicted_rating']]
    
    # Get top movies liked by this user (rating >= 4.0)
    liked_movies = user_ratings[user_ratings['rating'] >= 4.0][movie_col].values
    if len(liked_movies) == 0:
        liked_movies = user_ratings.sort_values(by='rating', ascending=False).head(3)[movie_col].values
        
    seen_movies = set(user_ratings[movie_col].values)
    recommendations = []
    
    # 2. For each liked movie, find similar ones
    for m_id in liked_movies:
        if m_id not in matrix.index:
            continue
            
        # Get nearest neighbors
        distances, indices = model.kneighbors(matrix.loc[m_id].values.reshape(1, -1), n_neighbors=top_n+1)
        
        for i in range(1, len(distances.flatten())):
            sim_movie_id = matrix.index[indices.flatten()[i]]
            if sim_movie_id not in seen_movies:
                # Score is based on similarity (1 - distance)
                score = 1 - distances.flatten()[i]
                recommendations.append((sim_movie_id, score))
    
    if not recommendations:
        return "🔍 No similar movies found based on your current ratings."
        
    # 3. Aggregate and rank
    rec_df = pd.DataFrame(recommendations, columns=[movie_col, 'score'])
    # Aggregate scores (take max similarity score across different liked movies)
    rec_df = rec_df.groupby(movie_col)['score'].max().sort_values(ascending=False).reset_index()
    rec_df = rec_df.head(top_n)
    
    # Merge with movie metadata
    result = rec_df.merge(movies, on=movie_col)
    result['predicted_rating'] = (result['score'] * 5.0).clip(1.0, 5.0) # Map to 1-5 scale
    
    return result[['title', 'genres', 'year', 'predicted_rating']]

if __name__ == "__main__":
    print("Testing Recommendation Logic...")
    recs = get_recommendations(user_id=1, top_n=5)
    print(recs)
