import os
import sys
import pandas as pd
import numpy as np

# Add src to path if needed for standalone execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from recommend import get_recommendations, load_assets
except ImportError:
    # If run from root, src might need to be added differently
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from recommend import get_recommendations, load_assets

def evaluate_model():
    print("Starting Model Evaluation...")
    assets = load_assets()
    if assets is None:
        print("Error: Could not load assets for evaluation.")
        return

    ratings = assets['ratings']
    user_col = assets['user_col']
    movie_col = assets['movie_col']
    
    # Select a sample of users for evaluation
    unique_users = ratings[user_col].unique()
    sample_size = min(50, len(unique_users))
    sample_users = unique_users[:sample_size]
    
    print(f"Evaluating {len(sample_users)} users (Efficiency: Assets pre-loaded)")
    
    hits = 0
    total = 0
    mae_sum = 0
    
    for user_id in sample_users:
        # Get ground truth: movies the user rated >= 4.0
        actual_liked_df = ratings[(ratings[user_col] == user_id) & (ratings['rating'] >= 4.0)]
        actual_liked_ids = set(actual_liked_df[movie_col])
        
        if not actual_liked_ids:
            continue
            
        # Get recommendations using pre-loaded assets
        recs = get_recommendations(user_id, top_n=10, assets=assets)
        
        if isinstance(recs, pd.DataFrame):
            # Hit Rate check
            rec_movie_ids = set(assets['matrix'].index[recs.index] if 'movieId' not in recs.columns else recs[movie_col])
            
            # Simplified check using titles to be safer across different ID formats
            rec_titles = set(recs['title'])
            actual_titles = set(assets['movies'][assets['movies'][movie_col].isin(actual_liked_ids)]['title'])
            
            if rec_titles.intersection(actual_titles):
                hits += 1
            
            # Simplified MAE (Difference between predicted rating and average actual rating of liked movies)
            avg_actual = actual_liked_df['rating'].mean()
            avg_pred = recs['predicted_rating'].mean()
            mae_sum += abs(avg_actual - avg_pred)
            
            total += 1
            
    if total > 0:
        hit_rate = (hits / total) * 100
        avg_mae = mae_sum / total
        print(f"\nEvaluation Results:")
        print(f"   Users evaluated: {total}")
        print(f"   Hit Rate @ 10:   {hit_rate:.2f}%")
        print(f"   Approx. MAE:     {avg_mae:.2f}")
        print("\n(Note: Hit Rate means at least one recommendation matched a movie the user liked in history)")
    else:
        print("No users found with enough ratings for evaluation.")

if __name__ == "__main__":
    evaluate_model()
