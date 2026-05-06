import streamlit as st
import sys
import os
import pandas as pd

# Add src to path to import recommend logic
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from recommend import get_recommendations, load_assets
except ImportError as e:
    st.error(f"❌ **Dependency Error:** {e}")
    st.info("💡 Make sure you have installed all requirements: `pip install -r requirements.txt`")
    st.stop()
except Exception as e:
    st.error(f"❌ **Error loading components:** {e}")
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Force Dark Theme visually and hide theme/github options
st.markdown("""
    <style>
    /* Global Background and Text */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
        color: #f8fafc !important;
    }

    /* Hide GitHub icon and Deploy button */
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    
    /* Hide System and Light theme options */
    [data-testid="stThemeProvider"] > div > button:nth-child(1),
    [data-testid="stThemeProvider"] > div > button:nth-child(2) {
        display: none !important;
    }
    [data-testid="stThemeProvider"] > div > div:nth-child(1),
    [data-testid="stThemeProvider"] > div > div:nth-child(2) {
        display: none !important;
    }

    footer {visibility: hidden;}
    
    /* Premium Header Styling */
    .premium-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        padding: 40px;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 40px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #ef4444 0%, #f87171 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px !important;
        letter-spacing: -1px;
    }
    .sub-title {
        color: #94a3b8 !important;
        font-size: 1.2rem !important;
        font-weight: 400 !important;
    }
    .movie-card {
        padding: 24px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 8px solid #ef4444;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .movie-card:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.2);
    }
    .rating-badge {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85em;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Premium Header Implementation
st.markdown("""
    <div class="premium-header">
        <h1 class="main-title">🎬 Smart Movie Recommender</h1>
        <p class="sub-title">Personalized Recommendations Powered by KNN & Scikit-Learn</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar Info
st.sidebar.title("🛠️ System Control")
st.sidebar.markdown("---")
st.sidebar.success("✅ TensorFlow Removed")
st.sidebar.success("✅ Scikit-Learn Ready")
st.sidebar.success("✅ Optimized for Cloud")

# Data Availability Check
assets = load_assets()

if assets is None:
    st.error("⚠️ **Data Files Not Found!**")
    st.warning("Please upload `movies.csv` and `ratings.csv` into the `data/` folder.")
    st.markdown("Example structure:")
    st.code("movie_recommender/\n├── data/\n│   ├── movies.csv\n│   └── ratings.csv\n└── app/app.py")
    st.stop()

# Dashboard
col1, col2 = st.columns([1, 2.5])

with col1:
    st.markdown("### ⚙️ Parameters")
    user_id = st.number_input("👤 User ID", min_value=1, value=1, step=1, help="Select a User ID to get custom picks.")
    top_n = st.slider("📊 Recommendations Count", 5, 20, 10)
    
    if st.button("🚀 Generate Picks", use_container_width=True):
        with st.spinner("🧠 Analyzing movie patterns..."):
            results = get_recommendations(user_id, top_n, assets=assets)
            
            if isinstance(results, str):
                st.warning(results)
            else:
                st.session_state['recommendations'] = results
                st.session_state['last_user'] = user_id

with col2:
    if 'recommendations' in st.session_state:
        recs = st.session_state['recommendations']
        target_user = st.session_state.get('last_user', user_id)
        
        st.markdown(f"### 🍿 Top Picks for User {target_user}")
        
        for idx, row in recs.iterrows():
            with st.container():
                st.markdown(f"""
                    <div class="movie-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; font-size: 1.4em;">{row['title']}</h3>
                            <span class="rating-badge">Match: {row['predicted_rating']:.1f}/5.0</span>
                        </div>
                        <p style="margin: 12px 0 6px 0; color: #94a3b8; font-size: 0.95em;">
                            <b>📅 Release Year:</b> {row['year']}
                        </p>
                        <p style="margin: 0; color: #94a3b8; font-size: 0.95em;">
                            <b>🎭 Genres:</b> {row['genres']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👈 Enter a **User ID** and click the button to see your personalized recommendations.")
        
        # Display Stats
        st.markdown("---")
        st.markdown("#### 📊 Dataset Overview")
        s1, s2, s3 = st.columns(3)
        s1.metric("Movies", len(assets['movies']))
        s2.metric("Ratings", len(assets['ratings']))
        s3.metric("Users", assets['ratings'][assets['user_col']].nunique())

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.9em;'>Built with ❤️ using Scikit-Learn | Fully Deployable on Streamlit Cloud</p>", unsafe_allow_html=True)
