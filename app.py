import os
import datetime
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from train import TextPreprocessor
except ImportError:
    
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
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

st.set_page_config(
    page_title="Advanced Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Gradient headers */
    .main-title {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        color: #576574;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    /* Prediction Cards */
    .prediction-card-real {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white !important;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(56, 239, 125, 0.25);
        font-weight: 800;
        font-size: 1.7rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-card-fake {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        color: white !important;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(239, 71, 58, 0.25);
        font-weight: 800;
        font-size: 1.7rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .report-preview {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #2a5298;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
    model_path = "models/best_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    return None, None

model, vectorizer = load_ml_assets()

st.sidebar.markdown("<h2 style='text-align: center; color: #1e3c72; font-weight: 800;'>NLP Engine</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.9rem; color: #7f8c8d;'>Fake News Detection Platform</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["Dashboard", "Model Performance", "About Project"])

# Sidebar details
st.sidebar.markdown("---")
st.sidebar.markdown("### Technology Stack")
st.sidebar.markdown("""
- **Backend:** Python, Scikit-learn
- **NLP Pipeline:** NLTK, Lemmatization
- **Serialization:** Joblib
- **Frontend:** Streamlit
- **Visuals:** Seaborn, Matplotlib
""")

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align: center; font-size: 0.8rem; color: #bdc3c7;'>Senior AI Architect Portfolio Project</p>", unsafe_allow_html=True)

if page == "Dashboard":
    st.markdown("<h1 class='main-title'>Advanced Fake News Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Analyze news articles in real-time using machine learning classifiers and natural language processing.</p>", unsafe_allow_html=True)

    if model is None or vectorizer is None:
        st.warning("⚠️ Model and TF-IDF Vectorizer files not found in the `models/` directory.")
        st.info("💡 Please run the training pipeline first: `python train.py` to train the models and generate resources.")
    else:
        
        st.markdown("### Enter News Article Text")
        article_text = st.text_area(
            "Paste the full content of the news article below for classification:",
            height=250,
            placeholder="Type or paste article content here..."
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("🚀 Analyze Article", use_container_width=True)

        if analyze_button or article_text:
            if not article_text.strip():
                st.error("Please enter some text before analyzing.")
            else:
                with st.spinner("Processing article text & running classification..."):
                   
                    preprocessor = TextPreprocessor()
                    clean_input = preprocessor.preprocess(article_text)
                    
                    if not clean_input.strip():
                        st.error("Preprocessing failed. The entered text does not contain enough english words.")
                    else:
                        
                        vec_input = vectorizer.transform([clean_input])
                        
                        pred = model.predict(vec_input)[0]
                        probs = model.predict_proba(vec_input)[0]  # [Prob(Fake), Prob(Real)]
                        
                        confidence = probs[pred] * 100
                    
                        st.markdown("### Analysis Results")
                        res_col1, res_col2 = st.columns([2, 2])
                        
                        with res_col1:
                            if pred == 1:
                                st.markdown("<div class='prediction-card-real'>VERDICT: REAL ARTICLE</div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div class='prediction-card-fake'>VERDICT: FAKE / UNRELIABLE</div>", unsafe_allow_html=True)
                            
                            m_col1, m_col2 = st.columns(2)
                            with m_col1:
                                st.markdown(f"<div class='metric-value'>{confidence:.1f}%</div><div class='metric-label'>Confidence</div>", unsafe_allow_html=True)
                            with m_col2:
                                char_cnt = len(article_text)
                                word_cnt = len(article_text.split())
                                st.markdown(f"<div class='metric-value'>{word_cnt}</div><div class='metric-label'>Word Count</div>", unsafe_allow_html=True)
                                
                        with res_col2:
                            st.markdown("#### Probability Distribution")
                            # Generate a clean Seaborn barplot for probabilities
                            fig, ax = plt.subplots(figsize=(6, 2.5))
                            categories = ['Fake News', 'Real News']
                            colors = ['#ff4d4d', '#2ecc71']
                            
                            sns.barplot(x=probs * 100, y=categories, palette=colors, ax=ax)
                            ax.set_xlim(0, 100)
                            ax.set_xlabel('Probability (%)')
                            ax.set_ylabel('')
                            sns.despine(left=True, bottom=True)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        
                        st.markdown("---")
                        st.markdown("### 📥 Download Prediction Report")
                        
                        verdict_str = "REAL" if pred == 1 else "FAKE"
                        report_text = f"""==================================================
FAKE NEWS DETECTION SYSTEM - ANALYSIS REPORT
==================================================
Timestamp:          {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Verdict:            {verdict_str}
Confidence Score:   {confidence:.2f}%
Model Used:         {type(model).__name__} (Calibrated)
Feature Vectorizer: TF-IDF (Term Frequency - Inverse Document Frequency)

TEXT METRICS:
- Character Count:   {char_cnt}
- Word Count:        {word_cnt}

PREPROCESSED TEXT SNIPPET:
"{clean_input[:300]}..."

ORIGINAL TEXT SNIPPET:
"{article_text[:300]}..."

==================================================
AI Architect Fake News Detection Platform
==================================================
"""
                        
                        st.markdown("#### Preview Report")

                        st.text_area(
                          "Report Preview",
                           report_text,
                           height=350,
                           disabled=True
                        )
                        
                        st.download_button(
                            label="Download PDF-Style Text Report",
                            data=report_text,
                            file_name=f"FakeNews_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )

elif page == "Model Performance":
    st.markdown("<h1 class='main-title'>Model Performance Leaderboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Compare cross-validated accuracy and test performance metrics across trained models.</p>", unsafe_allow_html=True)
    
    leaderboard_csv = "models/leaderboard.csv"
    if os.path.exists(leaderboard_csv):
        df_lb = pd.read_csv(leaderboard_csv)
        
        st.markdown("### Classifiers Leaderboard")
        st.dataframe(
            df_lb.style.highlight_max(subset=["CV Accuracy", "Test Accuracy", "F1 Score"], color="#d4edda")
            .format({"CV Accuracy": "{:.2%}", "Test Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"}),
            use_container_width=True
        )
        
        st.markdown("### Performance Comparison Graph")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_melt = df_lb.melt(id_vars="Model", value_vars=["CV Accuracy", "Test Accuracy"], var_name="Metric", value_name="Score")
        
        sns.barplot(x="Model", y="Score", hue="Metric", data=df_melt, palette=["#3498db", "#2ecc71"], ax=ax)
        ax.set_ylim(0.8, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Classifier Model")
        plt.title("CV Accuracy vs. Test Accuracy across Models", fontsize=12, fontweight='bold')
        st.pyplot(fig)
        plt.close()
        
    else:
        st.warning(" Leaderboard data not found in `models/leaderboard.csv`.")
        st.info(" Please complete the model training phase by running `python train.py` to see the leaderboard comparison.")

    st.markdown("---")
    st.markdown("### Exploratory Data Analysis Plots")
    
    plot_options = {
        "Class Distribution": "Screenshots/class_distribution.png",
        "News Length Distribution": "Screenshots/news_length_distribution.png",
        "Article Length Boxplot": "Screenshots/article_length_boxplot.png",
        "Pie Chart Label Distribution": "Screenshots/class_pie_chart.png",
        "Top Word Frequencies": "Screenshots/top_words_comparison.png",
        "WordCloud (Real News)": "Screenshots/wordcloud_real.png",
        "WordCloud (Fake News)": "Screenshots/wordcloud_fake.png"
    }
    
    selected_plot_name = st.selectbox("Select EDA Visualization to display:", list(plot_options.keys()))
    plot_path = plot_options[selected_plot_name]
    
    if os.path.exists(plot_path):
        st.image(plot_path, caption=selected_plot_name, use_container_width=True)
    else:
        st.info(f"Plot '{selected_plot_name}' image file not found. Ensure `train.py` has run completely.")

elif page == "About Project":
    st.markdown("<h1 class='main-title'>About Fake News Detector Project</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>An end-to-end Machine Learning and NLP system architecture overview.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    ###  Business Problem & Objective
    Misinformation and fake news have grown exponentially, impacting public discourse, elections, and global events. 
    This system provides a robust, scalable NLP pipeline to classify news articles as either **REAL** (verifiable, objective reporting) or **FAKE** (biased, fabricated, or misleading articles).
    
    ###  System Pipeline Architecture
    1. **Data Engineering:** Ingests the ISOT Dataset. True articles are crawled from Reuters.com; Fake articles are compiled from unreliable sites flagged by PolitiFact and Wikipedia.
    2. **Anti-Leakage Data Cleaning:** Real news articles contain crawler patterns like *"WASHINGTON (Reuters) -..."*. A naive model learns that *"Reuters"* implies Real. Our preprocessor strips these patterns before training, forcing the model to learn deep semantic patterns.
    3. **NLP Preprocessing:** Lowercases text, removes URLs, special characters, and numbers. Performs tokenization, English stopword removal, and WordNet Lemmatization.
    4. **Feature Engineering:** Computes TF-IDF vectorization with customized unigrams and bigrams, optimized via max features selection to prevent overfitting.
    5. **Classifier Models:** Trains Logistic Regression, Naive Bayes, Random Forest, and Linear SVM.
    6. **Model Selection:** Automatically selects the model with the highest test accuracy score for deployment.
    7. **Interactive Interface:** Serves predictions via a Streamlit web interface with real-time feedback.
    
    ###  Dataset Details
    - **Total Articles:** ~44,000 articles
    - **True.csv:** 21,417 articles (labeled `1`)
    - **Fake.csv:** 23,502 articles (labeled `0`)
    - **Balancing:** Balanced distribution, minimizing model bias toward one class.
    """)
