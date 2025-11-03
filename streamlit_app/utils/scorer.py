#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import requests
import nltk


from utils.parser import parse_and_extract 

# --- CONSTANT
THIN_CONTENT_THRESHOLD = 500
SIMILARITY_THRESHOLD = 0.75 
LABEL_MAPPING = {0: 'High', 1: 'Low', 2: 'Medium'} 

# --- DATA AND MODEL LOADING ---
@st.cache_resource
def load_assets():
    """Loads all necessary models and data, memoized for Streamlit."""
    try:
       
        print("Starting NLTK resource downloads...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt_tab', quiet=True) 
        
        
        model = joblib.load('models/quality_model.pkl')
        
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
     
        df_features = pd.read_csv('data/features.csv')
        
        
        df_features['embedding'] = df_features['embedding'].apply(lambda x: np.array(json.loads(x)))
        existing_embeddings = np.stack(df_features['embedding'].values)
        
        return model, embedding_model, df_features, existing_embeddings
    except Exception as e:
        
        st.error(f"Error loading assets. Check files in models/ and data/: {e}")
        return None, None, None, None

def analyze_url_stream(url, model, embedding_model, df_features, existing_embeddings):
    """The main analysis function for Streamlit, orchestrating all steps."""
    try:
        # 1. Scrape, Parse, and Extract all features
        body_text, wc, sc, fs, pol, subj = parse_and_extract(url)

        # 2. Prepare Feature Vector for ML Model
    
        new_features = pd.DataFrame([{
            'word_count': wc,
            'sentence_count': sc,
            'flesch_reading_ease': fs,
            'sentiment_polarity': pol,
            'sentiment_subjectivity': subj
        }])

        # 3. Quality Scoring
        prediction_encoded = model.predict(new_features)[0]
        quality_label = LABEL_MAPPING.get(prediction_encoded, "Unknown")
        is_thin = wc < THIN_CONTENT_THRESHOLD

        # 4. Duplicate Check
        new_embedding = embedding_model.encode([body_text])[0]
        similarities = cosine_similarity(new_embedding.reshape(1, -1), existing_embeddings)[0]
        
        similar_to = []
        for i, sim in enumerate(similarities):
            if sim > SIMILARITY_THRESHOLD:
                similar_to.append({
                    "url": df_features.iloc[i]['url'],
                    "similarity": round(sim, 4)
                })

        return {
            "url": url,
            "word_count": wc,
            "readability": round(fs, 2),
            "quality_label": quality_label,
            "is_thin": bool(is_thin),
            "similar_to": similar_to,
            "sentiment_polarity": round(pol, 2),
            "sentiment_subjectivity": round(subj, 2)
        }

    except requests.exceptions.Timeout as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}

