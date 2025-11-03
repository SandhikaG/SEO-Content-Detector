#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from textblob import TextBlob

def get_sentiment_features(text):
    """Calculates sentiment polarity and subjectivity using TextBlob (Advanced NLP Bonus)."""
    if not text or len(text) < 10:
        return 0.0, 0.0
    
    analysis = TextBlob(text)
    # Polarity: -1.0 (Negative) to 1.0 (Positive)
    # Subjectivity: 0.0 (Objective) to 1.0 (Subjective)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

