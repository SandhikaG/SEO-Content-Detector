#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import textstat
from utils.features import get_sentiment_features 

def parse_and_extract(url):
    """
    Scrapes a URL, parses the main content, and calculates technical SEO features.
    
    Returns: body_text, word_count, sentence_count, flesch_score, polarity, subjectivity
    """
    try:
        response = requests.get(url, timeout=10)
        
        # --- 1. Scrape and Parse ---
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Focus on main content areas to exclude boilerplate/nav
        main_content_tag = soup.find('article') or soup.find('main') or soup.find('body')
        
        # Ensure body_text is always a string, falling back to an empty string if tag is None
        body_text = main_content_tag.get_text(separator=' ', strip=True) if main_content_tag else ""
        
        # New: Ensure body_text is a string before attempting tokenization
        if not isinstance(body_text, str):
             body_text = str(body_text) 
        
        # Calculate word count
        words = word_tokenize(body_text)
        word_count = len(words)
        
    except requests.exceptions.Timeout:
        # Re-raise as an error dictionary to be caught by the Streamlit runner
        raise Exception("Request timed out after 10 seconds.")
    except Exception as e:
        # Re-raise as an error dictionary
        raise Exception(f"Failed to scrape or process URL content: {e}")

    if word_count == 0 or len(body_text.strip()) == 0:
        raise Exception("Content not found or failed to parse main body. Check URL or scraping logic.")

    # --- 2. Basic Feature Extraction ---
 
    sentence_count = textstat.sentence_count(body_text)
    flesch_score = textstat.flesch_reading_ease(body_text) if word_count > 10 else 100.0

    # --- 3. Advanced NLP Feature Extraction (from features.py) ---
    polarity, subjectivity = get_sentiment_features(body_text)

    return body_text, word_count, sentence_count, flesch_score, polarity, subjectivity


# In[ ]:




