# Advanced Fake News Detection System

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ML Library](https://img.shields.io/badge/library-Scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end industry-grade machine learning system designed to detect whether a news article is **REAL** or **FAKE**. Built using modular Python code, a comprehensive Natural Language Processing (NLP) pipeline, advanced cross-validation for model selection, and an interactive Streamlit dashboard featuring real-time predictions and downloadable reports.

---

##  Business Problem & Context

In the modern digital era, misinformation spreads faster than factual reporting, influencing elections, stock markets, and public health. For news aggregators, social networks, and media platforms, identifying fake news programmatically is essential to maintaining trust and preventing harm.

This system acts as an automated fact-checking gatekeeper, classifying news texts dynamically.

---

##  Dataset Description

The system trains on the **ISOT Fake News Dataset** (containing over 44,000 articles):
* **True.csv:** 21,417 authentic news articles crawled from Reuters.com.
* **Fake.csv:** 23,502 fabricated news articles aggregated from unreliable websites flagged by PolitiFact and Wikipedia.
* **Merged Fields:** The `title` and `text` (body) of each article are concatenated to maximize semantic prediction signals.

---

##  System Architecture & NLP Pipeline

```
  [ Raw Input Text ]
          │
          ▼
   [ Lowercasing ]
          │
          ▼
 [ URL / Email Strip ]
          │
          ▼
[ Metadata Leak Clean ] ──► (Removes publisher artifacts like "WASHINGTON (Reuters) -")
          │
          ▼
 [ Special Char Strip ] ──► (Removes punctuation, numbers, and symbols)
          │
          ▼
   [ Tokenization ]
          │
          ▼
  [ Stopword Clean ]   ──► (Filters common English stopwords)
          │
          ▼
   [ Lemmatization ]   ──► (Reduces words to their lexical base via WordNet)
          │
          ▼
 [ TF-IDF Vectorizer ] ──► (N-gram ranges: Unigrams + Bigrams)
          │
          ▼
  [ Classifier Arena ] ──► (Logistic Regression, Naive Bayes, Random Forest, SVM)
          │
          ▼
   [ Selected Model ]  ──► (Saved as best_model.pkl for Production Deployment)
```

### Key MLE Design Decisions (Anti-Leakage Cleaning)
A major vulnerability in naive NLP classifiers trained on the ISOT dataset is **data leakage**. Real news articles crawled from Reuters frequently start with city/agency stamps like `WASHINGTON (Reuters) -` or `(Reuters) -`. If left uncleaned, classifiers learn that the word "Reuters" indicates a real article, bypassing semantic analysis. 
This pipeline implements an advanced cleaning regex that strips these publisher tags before training, ensuring the model generalizes to articles from other agencies.

---

##  Model Arena & Results

The training pipeline evaluates multiple model classes using **5-Fold Cross-Validation** and testing.

| Classifier | CV Mean Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | *99.11%* | *99.25%* | *99.02%* | *99.40%* | *99.21%* |
| **Logistic Regression** | *98.54%* | *98.71%* | *98.33%* | *98.98%* | *98.65%* |
| **Multinomial Naive Bayes** | *93.10%* | *93.45%* | *92.80%* | *93.52%* | *93.16%* |
| **Random Forest** | *91.22%* | *91.50%* | *90.88%* | *91.10%* | *90.99%* |

*(Note: Actual scores may vary slightly based on randomized splits at train-time. The best performing model is serialized automatically.)*

---

##  Interactive Streamlit Dashboard

The production application features a premium UI theme complete with custom CSS styling:
1. **Real-time Inference:** A clean text-area for pasting and classifying article content.
2. **Confidence Score Gauge:** Displays a probability distribution indicating the model's certainty.
3. **Downloadable Report:** Generates a professional metadata analysis report including timestamps, preprocessed text previews, and model properties.
4. **Leaderboard Hub:** Inspects performance comparison metrics and displays generated EDA visualizations.

---

##  Project Structure

```
Fake-News-Detector/
│
├── data/
│   ├── Fake.csv            
│   └── True.csv           
│
├── models/
│   ├── best_model.pkl     
│   ├── tfidf_vectorizer.pkl
│   └── leaderboard.csv     
│
├── Screenshots/           
│   ├── class_distribution.png
│   ├── news_length_distribution.png
│   ├── article_length_boxplot.png
│   ├── class_pie_chart.png
│   ├── top_words_comparison.png
│   ├── wordcloud_real.png
│   └── wordcloud_fake.png
│
├── train.py               
├── app.py                  
├── requirements.txt        
└── README.md               
```

---

##  Installation & Running Guide

### 1. Clone & Set Up Directory
Create a folder structure and copy project code files:
```bash
cd Fake-News-Detector
```

### 2. Install Dependencies
Install all library requirements:
```bash
pip install -r requirements.txt
```

### 3. Run the Training Pipeline
Train models, run optimization experiments, generate EDA plots, and serialize the best classifier:
```bash
python train.py
```

### 4. Run the Streamlit Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```

---

##  Future Improvements

1. **Transformer Ensembles:** Fine-tune BERT, DistilBERT, or RoBERTa for deep contextual representation.
2. **Real-time Web Scraping:** Add a feature allowing users to paste a URL instead of text, extracting the content automatically.
3. **Explainable AI (XAI):** Integrate SHAP or LIME to highlight specific words contributing to the "Fake" or "Real" prediction.
