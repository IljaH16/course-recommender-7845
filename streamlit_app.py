import streamlit as st
import pandas as pd
from sklearn.decomposition import NMF
import os

# --- PATH LOGIC ---
# This finds the absolute path of the folder containing THIS script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Join that folder path with your filename
DATA_FILENAME = os.path.join(BASE_DIR, "pred_NMF_all.csv")

st.title("Course Recommender")

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Data Cleaning
        if 'user' in df.columns:
            df['user'] = pd.to_numeric(df['user'], errors='coerce')
            df = df.dropna(subset=['user'])
            # Convert to int to ensure number_input works correctly
            df['user'] = df['user'].astype(int)
        return df
    return None

df = load_data(DATA_FILENAME)

if df is not None:
    if 'user' in df.columns:
        # User Selection
        min_u = int(df['user'].min())
        max_u = int(df['user'].max())
        
        user_id = st.number_input(
            f"Enter User ID (available users {min_u} to {max_u}):", 
            min_value=min_u, 
            max_value=max_u, 
            value=min_u
        )
        
        if st.button("Find Recommendations"):
            # Filtering and sorting
            recs = df[df['user'] == user_id].sort_values(by='predicted_rating', ascending=False)
            
            if not recs.empty:
                st.write(f"### Top 10 Recommendations for User {user_id}")
                st.dataframe(recs.drop(columns=['user']).head(10))
            else:
                st.warning("No recommendations could be found. The user has yet a limited history.")
    else:
        st.error(f"Column 'user' not found in {os.path.basename(DATA_FILENAME)}")
else:
    # Diagnostic error to help you see where it's looking
    st.error(f"❌ File not found at: `{DATA_FILENAME}`")
    st.info(f"Files currently in this folder: {os.listdir(BASE_DIR)}")