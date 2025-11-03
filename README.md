# SEO Content Quality and Duplicate Detector

## Project Overview

This Streamlit application provides an instant analysis of content from any live URL. It combines core Natural Language Processing (NLP) features with a pre-trained machine learning model to classify content quality (High, Medium, or Low) and identifies potential near-duplicate content by checking semantic similarity against a baseline dataset. This tool is designed to help SEOs and content managers maintain high content standards and avoid cannibalization or duplication penalties.

##  Setup Instructions

To run this project locally, follow these steps:

**1.Clone the repository:**

git clone [https://github.com/SandhikaG/SEO-Content-Detector](https://github.com/SandhikaG/SEO-Content-Detector)

cd SEO-Content-Detector


**2.Install dependencies:** Ensure you have Python 3.9+ and install all necessary packages from the requirements file.

pip install -r requirements.txt


**3.Download NLP Data:** The NLTK and TextBlob libraries require language corpora data.

python -c "import nltk; nltk.download('punkt')"

python -m textblob.download_corpora


**4.Data Preparation & Model Training:**  The core assets (models/quality_model.pkl and data/features.csv) are assumed to be generated. If you need to re-train the model or generate new embeddings, run the Jupyter notebook:

jupyter notebook notebooks/seo_pipeline.ipynb


## Quick Start (Local Execution)

Run the Streamlit application from the project's root directory:

python -m streamlit run streamlit_app/app.py


A browser window will automatically open, where you can paste any URL and start the analysis.

## Deployed Streamlit URL

(REQUIRED for claiming Streamlit bonus points)

The application is deployed live and can be accessed here:

[Insert your deployed Streamlit URL here]

 ## Key Decisions


**Choice of Libraries (Sentence-Transformers):** Used for generating robust, semantic embeddings, which are far more accurate than keyword-based methods (like TF-IDF) for detecting plagiarism and conceptual duplication.

**HTML Parsing Approach:** Scraping specifically targets content within `<article>`, `<main>`, or the general `<body>` tags to ensure feature calculation is based purely on editorial content, minimizing boilerplate text.

**Similarity Threshold Rationale:** A Cosine Similarity threshold of 0.75 was set. This prevents minor grammatical edits from escaping detection while allowing genuinely new sections or updates to be correctly identified as unique.

**Model Selection Reasoning (Random Forest):** The Random Forest Classifier was chosen for its high accuracy, stability, and crucial interpretability, effectively mapping the non-linear relationship between diverse NLP features to the categorical quality label.

## Results Summary



**Model Accuracy/F1 Score:** The trained Random Forest model achieved an overall accuracy of 88.0% and a weighted F1-Score of 86.91% on the validation dataset.

**Number of Duplicates Found:** The initial dataset analysis identified 71 duplicate pairs using a similarity threshold of 0.85.

**Sample Quality Scores:** High-quality content in the dataset typically showed a Flesch Reading Ease score averaging 60.6 and a Word Count exceeding 5900 words.

**Thin Content Identified:** 35.8% of the analyzed pages were flagged as thin content (Word Count < 500).

## Limitations

**Scraping Dynamic Content:** The current scraping method relies on static HTML parsing (requests and BeautifulSoup). It may fail to retrieve the full body content from complex, heavily JavaScript-dependent sites or Single Page Applications (SPAs).

**Quality Label Dependency:** The accuracy of the ML model is fundamentally reliant on the quality and consistency of the initial human-labeled training data used to define "High," "Medium," and "Low" content.

**Sentiment Granularity:** Sentiment analysis uses the basic TextBlob library. It provides general polarity and subjectivity but may struggle with highly contextual, industry-specific, or nuanced technical language.
