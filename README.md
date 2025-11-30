🛰️ Hyperlocal News Anomaly Detection and Source Attribution

🎓 Capstone Project — GUVI Data Science Program

📌 End-to-End NLP Pipeline
(Raw Articles → Cleaning → Embedding → Sentiment + Location Extraction → Visualization → Model Training & Evaluation)

               ┌────────────────────────────────────────┐
               │             Data Ingestion             │
               │ (News Articles, Metadata, Location)    │
               └────────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │              Preprocessing Pipeline                │
        │  Cleaning ▪ Tokenization ▪ NER ▪ Geo-resolving     │
        │  Sentiment ▪ Topic Modeling ▪ Embeddings           │
        └────────────────────────────────────────────────────┘
                               │
                               ▼
      ┌────────────────────────────────────────────────────────┐
      │                    Feature Store                       │
      │  text_embed | sentiment | topics | geolocation | time  │
      └────────────────────────────────────────────────────────┘
                               │                         
                               ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        Anomaly Detection Core                       │
    │  ● Linguistic Anomaly (IsolationForest / VAE)                       │
    │  ● Predicted vs Extracted Location (Location Model)                 │
    │  ● Temporal Deviations (Prophet / ARIMA)                            │
    └─────────────────────────────────────────────────────────────────────┘ 
                               │                     
                               ▼
              ┌───────────────────────────────────────┐
              │              Scoring API              │
              │   combined_anomaly_score, reason      │
              └───────────────────────────────────────┘
                               │
                               ▼
              ┌───────────────────────────────────────┐
              │        Visualization Dashboard        │
              │  Maps ▪ Trends ▪ Article Drill-Down   │
              └───────────────────────────────────────┘


check the link: http://34.180.52.225:8501/


🎯 Objective

Detects unusual or misleading patterns in local news articles using NLP models (BERT/RoBERTa) by analyzing language, sentiment, and location for source verification.

📂 Key Files 

CSVFile - Dataset containing hyperlocal news articles 

PDF - colabnotebook PDF, py file(All have same coding) 

Models- Cleaned data with derived sentiment, topics, anomalies and trained models 

app.py - Streamlit dashboard code 

Readme- Documentation file about the python notebook & Datasets 

requirement - Dependencies list

Docker - It is a recipe that tells Docker how to build an image for your app



🧩 Business Use Cases

Disinformation Detection: Identify misattributed or fake news. 

Hyperlocal Trend Monitoring: Detect sentiment and topic shifts in regions.

Automated Content Verification: Flag suspicious content automatically.

📚 Project Approach

1.Data Ingestion / Preprocessing Clean, lemmatize, and extract geolocations using NER. 

2.Location Extraction from Text Using NLP techniques to infer actual location from the article content. 

3.Semantic & Sentiment Analysis Using Semantic embedding and Sentiment analysis 

4.Visualization: Interactive Streamlit dashboard with anomaly summaries, sentiment charts, and article reading section.

5.Anomaly Detection: Apply Isolation Forest / XGBoost / ARIMA with anomaly detection. 

6.Report Results: Display anomaly scores and flagged articles interactively.


🧰 Technologies Used


🐍 Python – Core programming language. 

🤖 Transformers (BERT / RoBERTa) – Semantic embeddings and NLP analysis. 

📚 spaCy / NLTK – Text preprocessing, tokenization, and NER. 

📊 scikit-learn – Anomaly detection (Isolation Forest, clustering). 

🗄️ pandas / NumPy – Data handling and manipulation. 

🖥️ Streamlit – Interactive web app dashboard. 

📈 plotly / matplotlib / folium – Visualizations for trends, sentiment, and maps. 

🗺️ Geocoding APIs / Gazetteers – Mapping extracted locations to real-world coordinates. 

☁️ AWS / GCP Hosting – Deploying and hosting the application.



✅ Conclusion 
  This project provides a robust system for detecting anomalous patterns in hyperlocal news articles. Use the Home page to explore news anomalies, check sentiment trends, and monitor potential misattributions. The app leverages advanced NLP models and cloud hosting to provide real-time insights efficiently.



👤 About the Author 

M. Keerthana| Data Science Enthusiast 🎓 

GitHub: [Checkout the link here](https://github.com/Keerthana-Mathaiyan?tab=repositories)

LinkedIn: [Keerthana Mathaiyan](https://www.linkedin.com/in/keerthana-mathaiyan/)


🧾 License This project is developed for educational and research purposes only


Reference: Guvi Live-Class colabnotebook,scikit-learn.org , docs.streamlit.ioand documents from Python.org  
