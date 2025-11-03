#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import json
import os
# Import the required functions from the modular utility file (utils/scorer.py)
from utils.scorer import load_assets, analyze_url_stream 

st.set_page_config(
    page_title="SEO Content Quality Detector",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("SEO Content Quality and Duplicate Detector")
st.markdown("Enter a URL below to run a real-time analysis against the trained model and existing data corpus.")

# Load models and data once. This function is memoized by @st.cache_resource.
model, embedding_model, df_features, existing_embeddings = load_assets()

if model is None:
    st.error("Failed to load models and data. Ensure all assets are in the correct `models/` and `data/` directories and the paths in `scorer.py` are correct.")
else:
    # --- Input Field ---
    url_input = st.text_input(
        "URL to Analyze:",
        placeholder="e.g., https://www.example.com/new-article",
    )

    if st.button("Analyze Content", type="primary") and url_input:
        with st.spinner("Scraping, Extracting Features, and Scoring..."):
            
            # Run the core analysis using the function from the modular scorer
            analysis_result = analyze_url_stream(
                url_input, model, embedding_model, df_features, existing_embeddings
            )

        st.subheader("Analysis Results")
        
        if "error" in analysis_result:
            st.error(analysis_result["error"])
        else:
            # --- Display Core Quality Score ---
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Quality Label", analysis_result['quality_label'])
            col2.metric("Word Count", analysis_result['word_count'])
            col3.metric("Readability (Flesch)", analysis_result['readability'])
            col4.metric("Thin Content?", "Yes" if analysis_result['is_thin'] else "No")

            # --- Duplicate Check ---
            st.markdown("---")
            st.subheader("Duplicate/Similarity Check")

            similar_list = analysis_result['similar_to']
            
            if similar_list:
                st.warning(f" Found {len(similar_list)} potential duplicate match(es)!")
                
                # Format the similarity data into a readable table
                similarity_data = []
                for item in similar_list:
                    similarity_data.append({
                        "URL": item['url'],
                        "Similarity Score": item['similarity']
                    })
                
                df_sim = pd.DataFrame(similarity_data)
                st.dataframe(df_sim, use_container_width=True)

            else:
                # Use SIMILARITY_THRESHOLD from scorer.py (implicitly 0.75)
                st.success("No near-duplicate content found in the existing corpus.")
                
            # --- Detailed Features (Bonus) ---
            st.markdown("---")
            st.subheader("Advanced Feature Breakdown (Including Sentiment)")
            
            # Displaying the full structured result, which includes sentiment and other features
            st.json(analysis_result)

st.caption("Deployment strategy: Streamlit App | Built using Random Forest and Sentence Transformers.")

