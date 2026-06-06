import os
import re
import sys
import joblib
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

DATA_DIR = "data"
MODELS_DIR = "models"
SCREENSHOTS_DIR = "Screenshots"
TRUE_DATA_URL = "https://media.githubusercontent.com/media/caesarw0/news-dataset/main/data/raw_data/True.csv"
FAKE_DATA_URL = "https://media.githubusercontent.com/media/caesarw0/news-dataset/main/data/raw_data/Fake.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def download_file(url, destination):
   
    if os.path.exists(destination):
        if os.path.getsize(destination) < 1024 * 1024:
            print(f"[Data Engineering] Existing file {destination} is too small (< 1MB). Removing LFS pointer.")
            os.remove(destination)
        else:
            print(f"[Data Engineering] {destination} already exists and is valid. Skipping download.")
            return
    print(f"[Data Engineering] Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        downloaded = 0
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    sys.stdout.write(f"\r  Progress: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.1f}%)")
                else:
                    sys.stdout.write(f"\r  Progress: {downloaded / (1024*1024):.1f}MB downloaded")
                sys.stdout.flush()
        print(f"\n[Data Engineering] Saved file to {destination}")
    except Exception as e:
        print(f"\n[Error] Failed to download {url}: {e}")
        print("[Warning] Please manually place True.csv and Fake.csv inside the 'data/' directory.")
        sys.exit(1)

def load_and_preprocess_data():
    true_path = os.path.join(DATA_DIR, "True.csv")
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    
    download_file(TRUE_DATA_URL, true_path)
    download_file(FAKE_DATA_URL, fake_path)
    
    print("[Data Engineering] Loading CSV files...")
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    
    df_true['label'] = 1
    df_fake['label'] = 0
    
    df = pd.concat([df_true, df_fake], ignore_index=True)
    print(f"[Data Engineering] Loaded {len(df_true)} true articles and {len(df_fake)} fake articles.")
    
    initial_len = len(df)
    df.drop_duplicates(subset=['title', 'text'], inplace=True)
    duplicates_removed = initial_len - len(df)
    print(f"[Data Engineering] Removed {duplicates_removed} duplicate articles.")
    
    initial_len = len(df)
    df.dropna(subset=['title', 'text'], inplace=True)
    nan_removed = initial_len - len(df)
    print(f"[Data Engineering] Removed {nan_removed} rows with missing text values.")
    
    df = df[df['text'].str.strip().str.split().apply(len) > 5]
    print(f"[Data Engineering] Validated dataset. Final shape: {df.shape}")
    
    print("\n" + "="*50)
    print("DATASET SUMMARY REPORT")
    print("="*50)
    print(f"Shape:            {df.shape}")
    print(f"Data Types:\n{df.dtypes.to_string()}")
    print("-"*50)
    print(f"Missing Values:\n{df.isnull().sum().to_string()}")
    print("-"*50)
    print(f"Class Balance:\n{df['label'].value_counts().to_string()}")
    print(f"Class Balance %:\n{(df['label'].value_counts(normalize=True) * 100).to_string()}")
    print("="*50 + "\n")
    
    return df


class TextPreprocessor:
    def __init__(self):
        self._ensure_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def _ensure_nltk_resources(self):
        resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
        for res in resources:
            try:
                if res in ['punkt', 'punkt_tab']:
                    nltk.data.find(f"tokenizers/{res}")
                else:
                    nltk.data.find(f"corpora/{res}")
            except LookupError:
                print(f"[NLTK] Downloading resource '{res}'...")
                nltk.download(res, quiet=True)
                
    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'^[a-z\s]+ \((reuters|reuters)\) - ', '', text)
        text = re.sub(r'^\(reuters\) - ', '', text)
        
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        tokens = word_tokenize(text)
        
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return " ".join(cleaned_tokens)


def generate_eda_plots(df, output_dir=SCREENSHOTS_DIR):
    print("[EDA] Generating professional visualizations...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='label', data=df, palette=['#ff4d4d', '#2ecc71'])
    plt.title("Class Distribution (0 = FAKE, 1 = REAL)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Number of Articles", fontsize=12)
    ax.set_xticklabels(['FAKE (0)', 'REAL (1)'])
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=10, fontweight='semibold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
    plt.close()
    
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='word_count', hue='label', element='step', stat='density', common_norm=False, kde=True, palette=['#ff4d4d', '#2ecc71'], log_scale=True)
    plt.title("Distribution of Article Word Counts (Log Scale)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Word Count (Log Scale)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "news_length_distribution.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='label', y='word_count', data=df, palette=['#ff4d4d', '#2ecc71'])
    plt.title("Article Word Count Boxplot Comparison", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Word Count", fontsize=12)
    plt.xticks([0, 1], ['FAKE (0)', 'REAL (1)'])
    plt.ylim(0, df['word_count'].quantile(0.95)) 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "article_length_boxplot.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    class_counts = df['label'].value_counts()
    plt.pie(class_counts, labels=['FAKE (0)', 'REAL (1)'], autopct='%1.1f%%', colors=['#ff4d4d', '#2ecc71'], startangle=90, explode=(0.05, 0), shadow=True, textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title("Label Distribution Percentage", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_pie_chart.png"), dpi=150)
    plt.close()

    fake_text_all = " ".join(df[df['label'] == 0]['clean_text'].astype(str))
    real_text_all = " ".join(df[df['label'] == 1]['clean_text'].astype(str))
    
    print("[EDA] Building wordcloud for Fake News...")
    wc_fake = WordCloud(width=800, height=450, background_color='black', colormap='Reds', max_words=100, random_state=42).generate(fake_text_all)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc_fake, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud - FAKE NEWS Corpus", fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wordcloud_fake.png"), dpi=150)
    plt.close()
    
    print("[EDA] Building wordcloud for Real News...")
    wc_real = WordCloud(width=800, height=450, background_color='black', colormap='Greens', max_words=100, random_state=42).generate(real_text_all)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc_real, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud - REAL NEWS Corpus", fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wordcloud_real.png"), dpi=150)
    plt.close()
    
    fake_words = fake_text_all.split()
    real_words = real_text_all.split()
    
    fake_counter = Counter(fake_words).most_common(20)
    real_counter = Counter(real_words).most_common(20)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    fake_words_df = pd.DataFrame(fake_counter, columns=['Word', 'Count'])
    sns.barplot(x='Count', y='Word', data=fake_words_df, ax=axes[0], palette='Reds_r')
    axes[0].set_title("Top 20 Most Frequent Words in FAKE News", fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel("Frequency", fontsize=12)
    axes[0].set_ylabel("")
    
    real_words_df = pd.DataFrame(real_counter, columns=['Word', 'Count'])
    sns.barplot(x='Count', y='Word', data=real_words_df, ax=axes[1], palette='Greens_r')
    axes[1].set_title("Top 20 Most Frequent Words in REAL News", fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel("Frequency", fontsize=12)
    axes[1].set_ylabel("")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_words_comparison.png"), dpi=150)
    plt.close()
    
    print(f"[EDA] Visualizations generated and saved inside '{output_dir}/'")


def experiment_tfidf(X_train_clean, y_train):
    print("\n" + "="*50)
    print("PHASE 4: TF-IDF VECTORIZATION OPTIMIZATION EXPERIMENTS")
    print("="*50)
    
    experiments = [
        {"ngram_range": (1, 1), "max_features": 5000, "name": "Unigrams (5k features)"},
        {"ngram_range": (1, 2), "max_features": 5000, "name": "Unigrams+Bigrams (5k features)"},
        {"ngram_range": (1, 2), "max_features": 10000, "name": "Unigrams+Bigrams (10k features)"}
    ]
    
    best_config = None
    best_score = 0.0
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    sample_size = min(15000, len(X_train_clean))
    indices = np.random.choice(len(X_train_clean), sample_size, replace=False)
    X_sample = [X_train_clean[i] for i in indices]
    y_sample = y_train.iloc[indices]
    
    for exp in experiments:
        vectorizer = TfidfVectorizer(ngram_range=exp["ngram_range"], max_features=exp["max_features"])
        X_vec = vectorizer.fit_transform(X_sample)
        
        clf = LogisticRegression(max_iter=500, random_state=42)
        scores = cross_val_score(clf, X_vec, y_sample, cv=kf, scoring='accuracy', n_jobs=-1)
        mean_score = np.mean(scores)
        
        print(f"Config: {exp['name']}")
        print(f"  CV Mean Accuracy: {mean_score:.4f} (+/- {np.std(scores):.4f})")
        
        idf_vals = vectorizer.idf_
        print(f"  IDF stats: Min={idf_vals.min():.4f}, Max={idf_vals.max():.4f}, Mean={idf_vals.mean():.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_config = exp
            
    print(f"\n[Feature Engineering] Selected Best Config: {best_config['name']} with {best_score*100:.2f}% proxy accuracy.")
    print("="*50 + "\n")
    return best_config


def main():
    print("="*60)
    print("ADVANCED FAKE NEWS DETECTION ML PIPELINE")
    print("="*60)
    
    df = load_and_preprocess_data()
    
    print("[Data Engineering] Concatenating title and body text...")
    df['full_text'] = df['title'] + " " + df['text']
    
    preprocessor = TextPreprocessor()
    
    df['clean_text'] = preprocess_texts(df['full_text'].tolist(), preprocessor)
    
    df = df[df['clean_text'].str.strip() != ""]
    print(f"[NLP Preprocessing] Finished. Valid clean shape: {df.shape}")
    
    generate_eda_plots(df)
    
    X = df['clean_text'].tolist()
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"[Pipeline] Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    best_config = experiment_tfidf(X_train, y_train)
    
    print("[Feature Engineering] Fitting final TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(ngram_range=best_config["ngram_range"], max_features=best_config["max_features"])
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    print("[Model Training] Defining classifiers...")
    
    svm_calibrated = CalibratedClassifierCV(estimator=LinearSVC(dual=False, random_state=42), cv=3)
    
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
        "Linear SVM": svm_calibrated
    }
    
    leaderboard = []
    trained_models = {}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, clf in classifiers.items():
        print(f"\n[Model Training] Training {name}...")
        
        cv_scores = cross_val_score(clf, X_train_vec, y_train, cv=kf, scoring='accuracy', n_jobs=-1)
        print(f"  CV Accuracies: {cv_scores}")
        print(f"  CV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        clf.fit(X_train_vec, y_train)
        trained_models[name] = clf
        
        y_pred = clf.predict(X_test_vec)
        test_acc = accuracy_score(y_test, y_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Test Precision: {precision:.4f}")
        print(f"  Test Recall:    {recall:.4f}")
        print(f"  Test F1 Score:  {f1:.4f}")
        
        leaderboard.append({
            "Model": name,
            "CV Accuracy": np.mean(cv_scores),
            "Test Accuracy": test_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
        
    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df.sort_values(by="Test Accuracy", ascending=False, inplace=True)
    leaderboard_df.reset_index(drop=True, inplace=True)
    
    print("\n" + "="*80)
    print("MODEL LEADERBOARD & COMPARISON")
    print("="*80)
    print(leaderboard_df.to_string(index=True))
    print("="*80 + "\n")
    
    best_model_name = leaderboard_df.iloc[0]["Model"]
    best_model_acc = leaderboard_df.iloc[0]["Test Accuracy"]
    print(f"[Model Persistence] Automatically selected the best model: '{best_model_name}' (Accuracy: {best_model_acc*100:.2f}%)")
    
    best_model = trained_models[best_model_name]
    
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    
    print(f"[Model Persistence] Saving best model to {best_model_path}...")
    joblib.dump(best_model, best_model_path)
    
    print(f"[Model Persistence] Saving TF-IDF Vectorizer to {tfidf_path}...")
    joblib.dump(tfidf, tfidf_path)
    
    leaderboard_df.to_csv(os.path.join(MODELS_DIR, "leaderboard.csv"), index=False)
    
    print("\n[Pipeline Complete] Models saved and project ready for Streamlit dashboard!")
    print("="*60)

def preprocess_texts(texts, preprocessor):
    cleaned = []
    total = len(texts)
    print(f"[NLP] Preprocessing {total} articles...")
    for i, text in enumerate(texts):
        cleaned.append(preprocessor.preprocess(text))
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1} / {total} articles ({(i + 1)/total * 100:.1f}%)")
    return cleaned

if __name__ == "__main__":
    main()
