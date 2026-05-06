# 🎬 Movie Recommendation System (KNN-Based)

This project implements a professional **Collaborative Filtering Recommender System** using Scikit-Learn. It utilizes the **K-Nearest Neighbors (KNN)** algorithm with Cosine Similarity to provide personalized movie recommendations based on user rating patterns.

## 🚀 Features

- **Efficient KNN Engine**: Uses Item-Item Collaborative Filtering for fast and relevant picks.
- **Robust Preprocessing**: Handles data cleaning, filtering, and feature extraction (year, genres) automatically.
- **Smart Caching**: Integrated with Streamlit's `cache_resource` for near-instant subsequent loads.
- **Premium UI**: A stunning, responsive Streamlit interface with a dark theme and interactive movie cards.
- **Cloud Ready**: Lightweight architecture with minimal dependencies, perfect for Streamlit Cloud deployment.

## 📁 Project Structure

```text
movie_recommender/
├── data/               # MovieLens dataset (ratings.csv, movies.csv)
├── app/                # Streamlit interface
│   └── app.py          # Main application entry point
├── src/                # Core logic
│   ├── recommend.py     # Main recommendation engine (KNN)
│   ├── evaluate.py      # Performance metrics (Hit Rate, MAE)
│   ├── preprocessing.py # Data cleaning utilities
│   ├── model.py         # [Deprecated]DL model placeholder
│   └── train.py         # [Deprecated]DL training placeholder
├── requirements.txt    # Dependencies (Pandas, Scikit-Learn, Streamlit)
└── README.md           # Documentation
```

## 🧠 Methodology

### 1. Item-Item Collaborative Filtering
The system builds a sparse matrix of Movie-User ratings. It then calculates the **Cosine Similarity** between movies. When a user likes a movie, the system finds the "nearest neighbors" in the high-dimensional rating space to suggest similar titles.

### 2. Cold Start Handling
If a User ID is not found in the dataset, the system falls back to recommending the most popular movies (highest global rating count) to ensure the user still gets a great experience.

### 3. Feature Extraction
- **Temporal Analysis**: Extracts release years from titles for better display and filtering.
- **Genre Metadata**: Displays movie genres and match scores for transparency.

## 🛠️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app/app.py
```

### 3. Evaluate Results
To see the system's performance metrics:
```bash
python src/evaluate.py
```

## 📊 Results

The KNN-based system provides highly relevant "More Like This" recommendations with a focus on high-rated content.
- **Hit Rate @ 10**: Measures if the top 10 picks contain at least one movie the user previously liked.
- **Efficiency**: No training time required; fits and caches in seconds.

---
Built with ❤️ using Scikit-Learn | Optimized for Production.
