import os
import json
import pickle
import warnings
import joblib
import requests

import io
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import torch
import xgboost as xgb

import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer 
import streamlit as st
from streamlit_option_menu import option_menu

warnings.filterwarnings("ignore")



print('Library loaded successfully')

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="🛰️Hyperlocal News Anomaly Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)
print('Page configuration setup successfully')

# ==========================
# SIDEBAR NAVIGATION
# ==========================

#==========================Menu =======================================#
with st.sidebar:
    selected = option_menu(
        "Main Menu", ["📊Project Insights ", "🔍 Anomaly Detection", "About"],
        icons=["bar-chart", "image", "info-circle"], menu_icon="menu-up", default_index=0
    )
print('menu updated successfully')


# -------------------------------
# 1️⃣ Load processed dataset
# -------------------------------
# Helper function to download and load a model from a URL
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(io.BytesIO(response.content))  # For pickle-based models
    else:
        print(f"Failed to load model from {url}. Status code: {response.status_code}")
        return None

# Function to download and load a PyTorch model from a URL
def load_pytorch_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Use io.BytesIO() to convert the response content into a file-like object
        return torch.load(io.BytesIO(response.content))
    else:
        print(f"Failed to download PyTorch model from {url}. Status code: {response.status_code}")
        return None

# Define the URLs for your models and data
xgb_Model_url = "https://storage.googleapis.com/anomalygithub/xgb_model.pkl"
log_reg_url = "https://storage.googleapis.com/anomalygithub/lr_model.pkl"
one_svm_url = "https://storage.googleapis.com/anomalygithub/one_class_svm_model.pkl"
Arima_url = "https://storage.googleapis.com/anomalygithub/arima_model.pkl"

tfidf_url = "https://storage.googleapis.com/anomalygithub/tfidf_vectorizer.pkl"
label_encoder_url = "https://storage.googleapis.com/anomalygithub/label_encoder.pkl"
DF_url = "https://storage.googleapis.com/anomalygithub/Anomaly_Detection.csv"
VAE_url = "https://storage.googleapis.com/anomalygithub/vae_model.pth"

# Load models from URLs using the helper function
xgb_model = load_model_from_url(xgb_Model_url)
one_class_svm_model = load_model_from_url(one_svm_url)
arima_model = load_model_from_url(Arima_url)


# Load the PyTorch model (VAE model) using the special function for PyTorch
vae_model = load_pytorch_model_from_url(VAE_url)

# Load Logistic Regression, TFIDF, and Label Encoder with joblib (for .pkl files)
tfidf = joblib.load(io.BytesIO(requests.get(tfidf_url).content))
le = joblib.load(io.BytesIO(requests.get(label_encoder_url).content))
lr_model = joblib.load(io.BytesIO(requests.get(log_reg_url).content))

print("Models loaded successfully")

# Load CSV from the URL into a pandas DataFrame
df = pd.read_csv(DF_url)
print("DataFrame loaded successfully")

df["Article"] = df["Article"].str.replace(r"\n", "", regex=True)
print(repr(df["Article"].iloc[0][:10]))


# ======================================================================
#                      Load Dataset (NO FILE UPLOAD)
# ======================================================================
# ---------------- Prediction function ----------------
def predict_place(text):
    X_input = tfidf.transform([text])
    # Logistic Regression
    prob_lr = lr_model.predict_proba(X_input)[0]
    pred_class_lr = np.argmax(prob_lr)
    pred_place_lr = le.inverse_transform([pred_class_lr])[0]
    conf_lr = prob_lr[pred_class_lr]

    # XGBoost
    prob_xgb = xgb_model.predict_proba(X_input)[0]
    pred_class_xgb = np.argmax(prob_xgb)
    pred_place_xgb = le.inverse_transform([pred_class_xgb])[0]
    conf_xgb = prob_xgb[pred_class_xgb]

    return (pred_place_lr, conf_lr), (pred_place_xgb, conf_xgb)
#====================================
#==========Page content==============
#====================================


if selected == "📊Project Insights ":
   
    st.title("📍📰 Hyperlocal News Anomaly Detection")
    # Create three main tabs
    tab_Project_Overview, tab_Visual_Analysis, tab_model_comparison = st.tabs([
        "Project Overview", "Visual Analysis", "Model Comparison"
    ])

    with tab_Project_Overview:
        # Title
        
        
        # -----------------------------------------------------------
        # Project Goals and Business Use Cases
        # -----------------------------------------------------------

        st.markdown("## 🎯 Project Goals")
        st.markdown("""
            Hyperlocal News Anomaly Detection identifies unusual spikes or patterns in neighborhood-level news by analyzing text, frequency, sentiment, and topic deviations.  
        The system detects emerging events such as **crime surges**, **local protests**, **accidents**, **weather disruptions**, and **civic issues** before they appear in mainstream media.
        """)

        st.markdown("## 🧩 Business Use Cases")
        st.markdown("""
        - **Disinformation Detection:** Helps media and government organizations identify articles where the reported location or source seems inconsistent with the content, potentially flagging misinformation or propaganda.  
        - **Hyperlocal Trend Monitoring:** Assists journalism and research agencies in tracking emerging topics, evolving sentiment, and developing events across specific geographic regions.  
        - **Automated Content Verification:** Enables news platforms and aggregators to automatically flag articles requiring human review due to unusual characteristics.  
        - **Brand & Reputation Management:** Supports businesses by monitoring hyperlocal news for anomalies related to their brand, detecting crises early and understanding location-specific sentiment.
        """)

        # -----------------------------------------------------------
        # Project Approach
        # -----------------------------------------------------------
        # Project Approach
        st.markdown("## 🔍 Project Approach")

        st.markdown("""
        ### **1. Data Processing 🧹**
        - Clean article and headline text.  
        - Extract and normalize locations using NER.  
        - Add sentiment scores and topic labels (BERTopic/LDA).  
        - Convert date fields into meaningful temporal features.

        ### **2. Feature Engineering 🧠📊**
        - Generate transformer-based embeddings for text.  
        - Combine embeddings with location, sentiment, topic, and temporal features.  
        - Build historical profiles to understand typical patterns for each location and news type.

        ### **3. Anomaly Detection ⚠️🤖**
        - **Linguistic anomalies:** Identify unusual writing or content using models such as Isolation Forest or One-Class SVM.  
        - **Source discrepancies:** Predict the likely origin of an article and compare it with the extracted location.  
        - **Temporal anomalies:** Detect sudden shifts or spikes in topics or sentiment using time-series methods.

        ### **4. Visualization & Alerts 📈🚨**
        - Integrate outputs from all anomaly detection components.  
        - Display anomaly scores, categories, and explanations through an interactive dashboard.
        """)

       
    with tab_Visual_Analysis:
      


        # ---------------------------
        # HEADER
        # ---------------------------
        
        st.write("Explore hyperlocal news using NLP, sentiment analysis, topics, and anomaly detection.")
        # -----------------------------
        # 1️⃣ Word Cloud of Cleaned Articles
        # -----------------------------
        st.header("1️⃣ Word Cloud - Cleaned Articles")
        text = " ".join(df["Article"].astype(str))

        wc = WordCloud(width=900, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # -----------------------------
        # 2️⃣ Word Cloud for Specific Location
        # -----------------------------
        st.header("2️⃣ Word Cloud by Country")

        country = st.selectbox("Pick a country", df["Country"].unique())

        text_loc = " ".join(df[df["Country"] == country]["Article"].astype(str))

        wc2 = WordCloud(width=900, height=400, background_color="white").generate(text_loc)
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.imshow(wc2, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)


        # -----------------------------
        # 3️⃣ Sentiment Distribution Across Top 8 Locations
        # -----------------------------
        st.header("3️⃣ Sentiment Distribution Across Top 8 Locations")
        top8 = df["Country"].value_counts().head(8).index
        df_top = df[df["Country"].isin(top8)]

        fig3 = px.box(
            df_top,
            x="Country",
            y="roberta_neg",
            color="Country",
            title="Negative Sentiment Distribution Across Top 8 Places"
        )
        st.plotly_chart(fig3)


        # -----------------------------
        # 4️⃣  Top 15 Places by Count
        # -----------------------------
        st.header("4️⃣ Top 15 Country by Article Count")
        top_Country = df["Country"].value_counts().head(15).reset_index()
        top_Country.columns = ["Country", "Count"]

        fig5 = px.bar(top_Country, x="Country", y="Count", title="Top 15 Country Mentioned")
        st.plotly_chart(fig5)

        # -----------------------------
        # 5️⃣ Geospatial Visualization of Place vs Article Count
        # -----------------------------
        st.header("5️⃣ Country-Based Clustering (Using Embeddings + Sentiment + Topics)")
       

        # ---------------------------------------------------------
        # SAFE EMBEDDING PARSER (handles spaces, missing commas, etc.)
        # ---------------------------------------------------------
        def parse_embedding(x):
            x = str(x).replace("\n", " ").strip("[]")
            return np.fromstring(x, sep=" ")

        embeddings = np.vstack(df["combined_embedding"].apply(parse_embedding))

        # ---------------------------------------------------------
        # BUILD FEATURES (X) AND LABELS (y)
        # ---------------------------------------------------------
        X = np.hstack([
            embeddings,
            df[["roberta_neg", "roberta_neu", "roberta_pos", "topic", "topic_prob"]].values
        ])
        y = df["Country"]   # <-- Country is your clustering target

        # ---------------------------------------------------------
        # TRAIN/TEST SPLIT
        # ---------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # ---------------------------------------------------------
        # STANDARDIZE FEATURES
        # ---------------------------------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ---------------------------------------------------------
        # LOGISTIC REGRESSION CLASSIFIER
        # ---------------------------------------------------------
        logreg = LogisticRegression(max_iter=500)
        logreg.fit(X_train_scaled, y_train)
        y_pred = logreg.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"### Logistic Regression Accuracy: **{acc:.3f}**")

        # ---------------------------------------------------------
        # PCA FOR CLUSTER VISUALIZATION
        # ---------------------------------------------------------
        pca = PCA(n_components=2)
        X_test_2d = pca.fit_transform(X_test_scaled)

        viz_df = pd.DataFrame({
            "pca1": X_test_2d[:, 0],
            "pca2": X_test_2d[:, 1],
            "ActualCountry": y_test.values,
            "PredictedCountry": y_pred
        })

        # ---------------------------------------------------------
        # COUNTRY CLUSTER IMAGE (Color=Predicted, Symbol=Actual)
        # ---------------------------------------------------------
        fig = px.scatter(
            viz_df,
            x="pca1",
            y="pca2",
            color="PredictedCountry",
            symbol="ActualCountry",
            title="Country Clustering Visualization Based on Article Content",
            hover_data=["ActualCountry", "PredictedCountry"]
        )

        st.plotly_chart(fig)


        # -----------------------------
        # 6️⃣Newstype vs Country
        # -----------------------------
        st.header("6️⃣ NewsType vs Country")

        group_df = df.groupby(["NewsType", "Country"]).size().reset_index(name="count")

        fig7 = px.bar(
            group_df,
            x="Country",
            y="count",
            color="NewsType",
            barmode="group",
            title="NewsType vs Country"
        )
        st.plotly_chart(fig7)

        # -----------------------------
        # 8️7️⃣ Pie Chart for News Category
        # -----------------------------
        st.header("7️⃣ News Category Breakdown")

        cat_df = df["News_Category"].value_counts().reset_index()
        cat_df.columns = ["Category", "Count"]

        fig8 = px.pie(
            cat_df,
            names="Category",
            values="Count",
            title="Distribution of News Categories"
        )
        st.plotly_chart(fig8)

        # -----------------------------
        # 8️⃣ Country Classification Model (Logistic Regression + XGBoost)
        # -----------------------------

        st.header("8️⃣ News Brand-Based Clustering (Using Embeddings + Sentiment + Topics)")

        # ---------------------------------------------------------
        # SAFE EMBEDDING PARSER
        # ---------------------------------------------------------
        def parse_embedding(x):
            x = str(x).replace("\n", " ").strip("[]")
            return np.fromstring(x, sep=" ")

        embeddings = np.vstack(df["combined_embedding"].apply(parse_embedding))

        # ---------------------------------------------------------
        # Features (X) + Labels (y = News Brand)
        # ---------------------------------------------------------
        X = np.hstack([
            embeddings,
            df[["roberta_neg", "roberta_neu", "roberta_pos", "topic", "topic_prob"]].values
        ])
 
        # If a news article mentions multiple brands, pick the first one
        y = df["brands_mentioned"].apply(lambda x: str(x).split(",")[0].strip())

        # ---------------------------------------------------------
        # Train/Test Split
        # ---------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # ---------------------------------------------------------
        # Standardize
        # ---------------------------------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ---------------------------------------------------------
        # Logistic Regression Classifier
        # ---------------------------------------------------------
        logreg = LogisticRegression(max_iter=500)
        logreg.fit(X_train_scaled, y_train)
        y_pred = logreg.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"### Logistic Regression Accuracy (News Brand): **{acc:.3f}**")

        # ---------------------------------------------------------
        # PCA for 2D Cluster Visualization
        # ---------------------------------------------------------
        pca = PCA(n_components=2)
        X_test_2d = pca.fit_transform(X_test_scaled)

        viz_df = pd.DataFrame({
            "pca1": X_test_2d[:, 0],
            "pca2": X_test_2d[:, 1],
            "ActualBrand": y_test.values,
            "PredictedBrand": y_pred
        })

        # ---------------------------------------------------------
        # Scatter Plot: Color = Predicted, Symbol = Actual
        # ---------------------------------------------------------
        fig = px.scatter(
            viz_df,
            x="pca1",
            y="pca2",
            color="PredictedBrand",
            symbol="ActualBrand",
            title="News Brand Clustering Based on Article Content",
            hover_data=["ActualBrand", "PredictedBrand"]
        )

        st.plotly_chart(fig)

    with tab_model_comparison:
        

        # ===========================
        # 1️⃣ Linguistic Anomaly Detection
        # ===========================
        st.header("Linguistic Anomaly Detection Model")

        st.subheader("🔹 Anomalies Detected")
        st.write("- iso_anomaly: 130 anomalies")
        st.write("- svm_anomaly: 194 anomalies")
        st.write("- vae_anomaly: 62 anomalies")
        st.write("- deep_svdd_anomaly: 126 anomalies")
        st.write("- Consensus anomalies (at least 2 models agree): 84")

        st.subheader("🔹 Model Overlap (Jaccard Similarity)")
        st.write("- iso_anomaly vs svm_anomaly → Jaccard: 0.161")
        st.write("- iso_anomaly vs vae_anomaly → Jaccard: 0.116")
        st.write("- iso_anomaly vs deep_svdd_anomaly → Jaccard: 0.024")
        st.write("- svm_anomaly vs vae_anomaly → Jaccard: 0.148")
        st.write("- svm_anomaly vs deep_svdd_anomaly → Jaccard: 0.029")
        st.write("- vae_anomaly vs deep_svdd_anomaly → Jaccard: 0.011")

        st.markdown("---")  # horizontal line

       
        # ===========================
        # 3️⃣ Temporal Anomaly Detection
        # ===========================
        st.header(" Temporal Anomaly")

        st.subheader("=== Supervised Evaluation ===")

        st.write("**Model:** zscore_anomaly")
        st.write("- ROC-AUC: nan")
        st.write("- PR-AUC: 0.5000")
        st.write("- F1-score (top 5% threshold): 0.0000")

        st.write("**Model:** arima_anomaly_score")
        st.write("- ROC-AUC: nan")
        st.write("- PR-AUC: 0.5000")
        st.write("- F1-score (top 5% threshold): 0.0000")




elif selected == "🔍 Anomaly Detection":
    st.title("🔍 Anomaly Detection")
       # Create three main tabs
    tab_LAD, tab_SDD, tab_Temporal_Anomaly_Detection = st.tabs([
        "Linguistic Anomaly Detection", "Source Discrepancy Detection", "Temporal Anomaly Detection"
    ])
    with tab_LAD:
           # ---------------------------
        # Select Heading
        # ---------------------------
        selected_heading = st.selectbox("Select an Article Heading", df["Heading"].unique())

        # Filter the selected article
        article_row = df[df["Heading"] == selected_heading].iloc[0]

        # ---------------------------
        # Overall Anomaly Highlight
        # ---------------------------
        anomaly_cols = ['iso_anomaly', 'svm_anomaly', 'vae_anomaly', 'deep_svdd_anomaly']
        is_anomaly = any(article_row[col] == 1 for col in anomaly_cols)

        if is_anomaly:
            st.markdown("<h2 style='color:red'>⚠️ This Article is Detected as Anomaly!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green'>✅ This Article is Normal</h2>", unsafe_allow_html=True)

        # ---------------------------
        # Model-Level Results
        # ---------------------------
        st.subheader("🔹 Model-Level Anomaly Detection")
        model_names = ["Isolation Forest", "SVM", "VAE", "Deep SVDD"]
        
        for model, col in zip(model_names, anomaly_cols):
            if article_row[col] == 1:
                st.markdown(f"<span style='color:red'>⚠️ {model}: Anomaly</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:green'>✅ {model}: Normal</span>", unsafe_allow_html=True)

        # ---------------------------
        # Article Info
        # ---------------------------
        st.subheader("📰 Article Information")
        st.markdown("**Article:**")
        st.text(article_row["Article"])


        st.markdown(f"**Date:** {article_row['Date']}")
       
        st.markdown(f"**Country:** {article_row['Country']}")
        st.markdown(f"**Brand(s) Mentioned:** {article_row['brands_mentioned']}")


        # ---------------------------
        # Sub Heading 1: Linguistic Anomaly Detection (Summary Lists)
        # ---------------------------
        st.subheader("📌 Linguistic Anomaly Summary (Model-wise)")

        # Dictionary mapping model names to their anomaly column
        model_to_col = {
            "Isolation Forest": "iso_anomaly",
            "SVM": "svm_anomaly",
            "VAE": "vae_anomaly",
            "Deep SVDD": "deep_svdd_anomaly"
        }

        # Loop through each model and create dropdowns
        for model_name, col in model_to_col.items():

            # Filter headings where the model detected an anomaly
            anomaly_headings = df[df[col] == 1]["Heading"].tolist()

            # Expandable section for each model
            with st.expander(f"{model_name} – Anomalous Headings ({len(anomaly_headings)})"):
                if len(anomaly_headings) == 0:
                    st.write("✅ No anomalies detected by this model.")
                else:
                    selected_heading_model = st.selectbox(
                        f"Select a heading flagged by {model_name}",
                        anomaly_headings,
                        key=col  # unique key for each dropdown
                    )

                    # Display the article for the selected anomaly
                    selected_row = df[df["Heading"] == selected_heading_model].iloc[0]
                    st.write("### Article Content")
                    st.text(selected_row["Article"])

    with tab_SDD:
       

        # ---------------- Streamlit UI ----------------
        # Dropdown for heading
        heading_list = df["Heading"].tolist()
        selected_heading = st.selectbox("Select a Heading", heading_list)

        # Extract article details
        row = df[df["Heading"] == selected_heading].iloc[0]

        # Predict
        (lr_pred, lr_conf), (xgb_pred, xgb_conf) = predict_place(row["Heading"])

        # Display predictions
        st.subheader("Prediction Results")
        st.write("### Logistic Regression")
        st.write(f"Predicted Place: **{lr_pred}** | Confidence: **{lr_conf:.3f}**")
        st.write("### XGBoost")
        st.write(f"Predicted Place: **{xgb_pred}** | Confidence: **{xgb_conf:.3f}**")

        # Article info
        st.subheader("Article Details")
        st.text(f"{row['Article']}")
        
        st.write(f"**Country:** {row['Country']}")

    with tab_Temporal_Anomaly_Detection:

        # Convert Date safely
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])

        # ---------------- Streamlit UI ----------------
        st.title("Location-based Sentiment & Anomaly Dashboard")

        # Select location
        locations = df['Place'].unique()
        selected_location = st.selectbox("Select a Location", locations)

        # Filter data
        df_loc = df[df['Place'] == selected_location]

        # ---------------- Prepare trend data ----------------
        df_trend = df_loc.groupby('Date').agg(
            article_count=('Article', 'count'),
            avg_sentiment=('sent_compound', 'mean'),
            zscore_anomaly=('zscore_anomaly', 'max'),          # assume 0/1 flag
            arima_anomaly=('arima_anomaly_score', 'mean')      # continuous score
        ).reset_index()

        # Threshold for ARIMA anomaly flag
        df_trend['arima_flag'] = df_trend['arima_anomaly'] > 0.8  # adjust threshold as needed

        # ---------------- Plot sentiment with anomalies ----------------
        fig = px.line(df_trend,
                      x='Date',
                      y='avg_sentiment',
                      title=f"Sentiment Trends for {selected_location}")

        # Add Z-score anomalies
        fig.add_scatter(
            x=df_trend[df_trend['zscore_anomaly'] == 1]['Date'],
            y=df_trend[df_trend['zscore_anomaly'] == 1]['avg_sentiment'],
            mode='markers',
            name='Z-score anomaly',
            marker=dict(color='red', size=10, symbol='x')
        )

        # Add ARIMA anomalies
        fig.add_scatter(
            x=df_trend[df_trend['arima_flag']]['Date'],
            y=df_trend[df_trend['arima_flag']]['avg_sentiment'],
            mode='markers',
            name='ARIMA anomaly',
            marker=dict(color='orange', size=10, symbol='triangle-up')
        )

        st.plotly_chart(fig)

        # ---------------- Optional: Topic trends ----------------
        # Group by Date and News_Category instead of topic
        df_category_trend = df_loc.groupby(['Date', 'News_Category']).agg(
            count=('Article', 'count')  # Or use 'topic_prob' if it makes sense in your data
        ).reset_index()

        # Plot News_Category trends over time
        fig_category = px.line(
            df_category_trend,
            x='Date',
            y='count',
            color='News_Category',
            title=f"News Category Trends Over Time for {selected_location}"
        )

        st.plotly_chart(fig_category)



elif selected == "About":
    st.title("📍📰 Hyperlocal News Anomaly Detection")
    st.markdown("### 🎓 Capstone Project — GUVI Data Science Program")
    st.markdown("## About the Project ")
    st.markdown("""
           A hyperlocal news anomaly detection system leveraging NLP and sentiment analysis to identify unusual patterns, spikes, and inconsistencies in neighborhood-level news.
        """)
    
    
    # -----------------------------------------------------------
    # Author Section
    # -----------------------------------------------------------
    st.markdown("## 👨‍💻 About the Author")

    st.markdown("""
    **Name:** *Keerthana🎓*  
    **Profile:** Data Science Enthusiast  
    **Career Goal:** Actively seeking opportunities in data science, machine learning, and AI-driven roles.  

    I am passionate about using machine learning, NLP, and analytical modeling to create impactful, real-world solutions.  
    This project was developed as part of my interest in applying AI to identify hyperlocal news anomalies and support information reliability.

   
    """)

    # Social Links Section
    st.subheader("🔗 Connect with Me")
    st.write("""
    **Keerthana| Data Science Enthusiast 🎓**   
    **GitHub:** [Checkout the link here](https://github.com/Keerthana-Mathaiyan?tab=repositories)  
    **LinkedIn:** [Click here](https://www.linkedin.com/in/keerthana-mathaiyan/)  
    """)

    # -----------------------------------------------------------
    # Tech Stack
    # -----------------------------------------------------------

    st.markdown("## 🧰 Tech Stack")

    st.markdown("""
    **💻 Language:**<br>
    🐍 Python 3.13.5<br><br>

    **🌐 Framework:**<br>
    Streamlit<br><br>

    **🛠 Models Used:**<br>
    ⚙️ Logistic Regression<br>
    ⚙️ Isolation Forest<br>
    ⚙️ One-Class SVM<br>
    ⚙️ Variational Autoencoder<br>
    ⚙️ Deep SVDD<br>
    ⚙️ XGBoost<br>
    ⚙️ Neural Network<br>
    ⚙️ ARIMA with anomaly detection<br>
    ⚙️ Statistical process control methods<br><br>

    **📦 NLP & Embeddings:**<br>
    📝 NLTK<br>
    💬 spaCy<br>
    🤗 Transformers<br>
    🧠 Sentence-Transformers<br>
    📊 BERTopic<br>
    ❤️ VADER Sentiment<br>
    🔥 Flair<br><br>

    **🌍 Geolocation & Fuzzy Matching:**<br>
    🌐 GeoPy<br>
    📍 GeoText<br>
    🗺 FlashGeoText<br>
    🌎 GeoNamesCache<br>
    🔤 FuzzyWuzzy<br>
    🇵🇾 PyCountry<br><br>

    **📈 Time Series & Forecasting:**<br>
    ⏱ Arima<br>
    """, unsafe_allow_html=True)
    # -----------------------------------------------------------
    # License Section
    # -----------------------------------------------------------

    st.markdown("## 📄 License")
    st.markdown("""
    This project is developed for **educational and research purposes only**.  
    All models, features, and methods are intended to demonstrate applied NLP and anomaly detection concepts.
    """)
    
    st.success("Thank you for exploring this project!")

    