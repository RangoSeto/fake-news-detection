# 📰 Fake News Detection System

An AI-powered Fake News Detection System that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) and Machine Learning techniques.

This project was developed as part of an Artificial Intelligence course to demonstrate the application of machine learning in misinformation detection.

---

## 📌 Project Overview

The rapid spread of misinformation on social media and online news platforms has become a major challenge worldwide. This project aims to automatically identify fake news articles by analyzing their textual content.

The system utilizes:

- Natural Language Processing (NLP)
- TF-IDF Feature Extraction
- Logistic Regression
- Random Forest
- Streamlit Web Application

Users can enter a news article, and the system predicts whether the content is **Real** or **Fake**.

---

## ✨ Features

- Clean and preprocess news articles
- Exploratory Data Analysis (EDA)
- TF-IDF text vectorization
- Machine Learning model training
- Model comparison
- Performance evaluation
- Interactive Streamlit web interface
- Real-time prediction

---

## 🗂 Project Structure

```
fake-news-detection/
│
├── app.py 
├── requirements.txt
├── 01_data_preprocessing.ipynb
├── 02_baseline_models.ipynb
├── README.md
│
├── data/
│   └── news.csv
│
├── Output/
│   ├── figures/
│   │   
│   │
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── tfidf_vectorizer.joblib
```


---

## 📊 Dataset

Dataset:

**Real and Fake News Dataset**

Source:
https://www.kaggle.com/datasets/nopdev/real-and-fake-news-dataset

The dataset contains thousands of labeled news articles classified as:

- Real News
- Fake News

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## 🧠 Machine Learning Models

This project compares two supervised learning algorithms:

| Model | Purpose |
|--------|----------|
| Logistic Regression | Baseline classifier |
| Random Forest | Ensemble learning model |

Text features are extracted using **TF-IDF Vectorization** before classification.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Lowercase conversion
- Removing punctuation
- Removing numbers
- Removing special characters
- Tokenization
- Stopword removal
- Lemmatization
- TF-IDF Vectorization

---

## 📈 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The comparison helps determine the most effective model for fake news classification.

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/RangoSeto/fake-news-detection.git

cd fake-news-detection
```

Create virtual environment (optional)

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Mac/Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 💻 How It Works

1. User enters a news article.
2. Text is preprocessed.
3. TF-IDF converts text into numerical features.
4. The trained machine learning model predicts whether the news is:
   - ✅ Real
   - ❌ Fake
5. The prediction is displayed in the Streamlit interface.


---

## ⚠️ Limitations

- The model is trained primarily on U.S. political news.
- Performance may decrease on news from different domains or countries.
- The system should be used as a decision-support tool rather than a definitive fact-checking solution.

---

## 👥 Authors

Developed as an Artificial Intelligence course project.

Team Name : *** The Truth Lens ***
Team Members:
Mae Nwe
Fixed
Aura
Rango
Izel
Mya Ei
Oak Kar
Serenity


---

## 📄 License

This project is for educational purposes.

Feel free to fork and improve it for learning and research.
