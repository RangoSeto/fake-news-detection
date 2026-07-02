"""Streamlit demo for fake news classification (CST-3121) with Conditional Word Explanations."""

from typing import Any
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import numpy as np

from preprocessing import build_full_text
from samples import SAMPLES

PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "Output" 
BERT_DIR = MODELS_DIR / "distilbert-best"
BERT_AVAILABLE = (BERT_DIR / "model.safetensors").exists()

LABEL_MAP = {0: "REAL", 1: "FAKE"}
DISTILBERT_LABEL = "DistilBERT"

CLASSICAL_MODELS = {
    "Logistic Regression": "logistic_regression_model.joblib",
    "Random forest": "random_forest_model.joblib",
}

MODEL_METRICS = {
    "Logistic Regression": {"accuracy": 0.932, "f1": 0.932, "split": "test"},
    "Random forest": {"accuracy": 0.906, "f1": 0.906, "split": "validation"},
    DISTILBERT_LABEL: {"accuracy": 0.968, "f1": 0.968, "split": "test"},
}


def available_models() -> list[str]:
    models = list(CLASSICAL_MODELS.keys())
    if BERT_AVAILABLE:
        models.append(DISTILBERT_LABEL)
    return models


@st.cache_resource
def load_classical_artifacts() -> tuple[Any, dict[str, Any]]:
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    models = {
        name: joblib.load(MODELS_DIR / filename)
        for name, filename in CLASSICAL_MODELS.items()
    }
    return vectorizer, models


@st.cache_resource
def load_bert_pipeline():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=str(BERT_DIR),
        tokenizer=str(BERT_DIR),
        device=-1,
        top_k=None,
    )


def predict_classical(vectorizer, model, full_text: str):
    X = vectorizer.transform([full_text])
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return pred, proba, X


def predict_bert(pipe, full_text: str):
    results = pipe(
        full_text,
        truncation=True,
        max_length=256,
    )[0]
    scores = {0: 0.0, 1: 0.0}
    for item in results:
        idx = int(item["label"].split("_")[1])
        scores[idx] = float(item["score"])
    pred = max(scores, key=scores.get)
    proba = [scores[0], scores[1]]
    return pred, proba


def explain_prediction(vectorizer, model, model_name, X_transformed):
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = X_transformed.nonzero()[1]
    if len(nonzero_indices) == 0:
        return None

    words_in_text = [feature_names[i] for i in nonzero_indices]
    tfidf_scores = [X_transformed[0, i] for i in nonzero_indices]

    df_words = pd.DataFrame({
        "Word": words_in_text,
        "TF-IDF": tfidf_scores
    })

    if model_name == "Logistic Regression":
        coefficients = model.coef_[0]
        word_weights = [coefficients[i] for i in nonzero_indices]
        df_words["Impact"] = df_words["TF-IDF"] * word_weights
        return df_words

    elif model_name == "Random forest":
        importances = model.feature_importances_
        word_weights = [importances[i] for i in nonzero_indices]
        df_words["Importance"] = df_words["TF-IDF"] * word_weights
        df_words = df_words.sort_values(by="Importance", ascending=False)
        return df_words

    return None


# =========================================================
#                         UI LAYOUT
# =========================================================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Updated CSS Injection (For modern Streamlit versions)
st.markdown("""
    <style>
    /* Main Content Width Tuning */
    .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: auto;
    }
    
    /* 1. Target the actual Tab Link Row Container directly */
    [data-baseweb="tab-list"] {
        gap: 25px !important; 
        display: flex !important;
    }
    
    /* 2. Target individual Tab Buttons precisely */
    [data-baseweb="tab"] {
        padding-left: 20px !important;
        padding-right: 20px !important;
        font-weight: 600 !important;
    }
    
    
    </style>
""", unsafe_allow_html=True)


st.title("📰 Fake News Detection System")
st.caption("CST-3121 Group Project")
st.write("---")

vectorizer, classical_models = load_classical_artifacts()
model_options = available_models()


tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home","🔍 Detector Workspace", "📖 About Project","⚙️ How It Works", "ℹ️ Disclaimer"])

# ----------------- TAB 1: HOME -----------------
with tab1:
    st.subheader("Welcome to the Fake News Detector")
    st.image(
        # "https://www.civilserviceworld.com/siteimg/news-main/ugc-1/fullnews/news/28932/31839_original.jpg",
        "https://as1.ftcdn.net/jpg/14/57/94/00/1000_F_1457940060_LqXqcaTJC08iCAAmB9PZfcxM4zyKA1LL.jpg", 
        # caption="Empowering Truth in the Digital Age"
    )
    st.markdown("""
    

    This application helps identify whether a news article is **Real** or **Fake**
    using Machine Learning techniques.

    Simply paste a news article into the **Detection** page and let the model
    analyze its content instantly.

    ---
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        ### 🎯 Features

        - Detect Real and Fake News
        - Confidence Score
        - Powered by Machine Learning
        - Fast prediction
        """)

    with col2:
        st.success("""
        ### 🤖 Models Used

        • Logistic Regression

        • Random Forest
        """)

    st.markdown("---")

    st.subheader("📌 Important")

    st.warning("""
    This system was trained using a dataset of **United States political news**.

    Predictions are expected to be more reliable for news articles within this
    domain. Results for entertainment, sports, finance, technology, health, or
    international news may be less accurate because these topics were not the
    primary focus of the training data.
    """)

with tab2:
    
    selectbox_help = (
        "DistilBERT achieved the best test accuracy (~96.8%) when trained locally."
        if BERT_AVAILABLE
        else "DistilBERT is not available until outputs/models/distilbert-best/ exists."
    )

    selected_model = st.selectbox(
        "Model",
        model_options,
        index=0,
        help=selectbox_help,
    )

    metrics = MODEL_METRICS[selected_model]
    st.caption(
        f"**Engine Configuration:** {selected_model} | "
        f"**Baseline Metric:** {metrics['accuracy']:.1%} accuracy ({metrics['split']} set)"
    )

    sample_choice = st.selectbox("Try a sample article", list(SAMPLES.keys()))

    if "prev_sample" not in st.session_state:
        st.session_state.prev_sample = sample_choice

    if sample_choice != st.session_state.prev_sample:
        if sample_choice != "— Select a sample —":
            sample = SAMPLES[sample_choice]
            if sample:
                st.session_state.title_input = sample["title"]
                st.session_state.body_input = sample["body"]
        st.session_state.prev_sample = sample_choice

    if "title_input" not in st.session_state:
        st.session_state.title_input = ""
    if "body_input" not in st.session_state:
        st.session_state.body_input = ""

    title = st.text_input("Article Headline / Title", key="title_input")
    body = st.text_area("Article Document Body Context", height=220, key="body_input")

    if body.strip():
        st.caption(f"📊 Tracking Matrix: {len(body.split())} tokens · {len(body)} raw characters")

    analyze = st.button("Execute Verification", type="primary", use_container_width=True)

    if analyze:
        if not title.strip() and not body.strip():
            st.error("Operation halted: Please provide either a headline or valid body text.")
        else:
            if len(body.strip()) < 20:
                st.warning("Warning: Prompt structure is extremely brief. Output fidelity might degrade.")

            full_text = build_full_text(title, body)

            if not full_text:
                st.error("Error: Input text yielded zero structural features after preprocessing transformations.")
            else:
                spinner_text = (
                    "Initializing DistilBERT tokenizers and performing inference..."
                    if selected_model == DISTILBERT_LABEL
                    else "Computing matrix vectorizations and executing prediction..."
                )
                with st.spinner(spinner_text):
                    X_transformed = None
                    if selected_model == DISTILBERT_LABEL:
                        bert_pipe = load_bert_pipeline()
                        pred, proba = predict_bert(bert_pipe, full_text)
                    else:
                        model = classical_models[selected_model]
                        pred, proba, X_transformed = predict_classical(vectorizer, model, full_text)

                label = LABEL_MAP[pred]
                confidence = proba[pred]

                # --- Results Visualization ---
                st.markdown("### 📊 Prediction Metrics")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if label == "REAL":
                        st.success(f"### 🟢 Result: **{label}**")
                    else:
                        st.error(f"### 🔴 Result: **{label}**")
                        
                with res_col2:
                    st.metric(label="Model Certainty Score", value=f"{confidence:.1%}")

                with st.expander("Probability Breakdown Vector", expanded=True):
                    st.progress(float(proba[0]), text=f"REAL Class Weight: {proba[0]:.1%}")
                    st.progress(float(proba[1]), text=f"FAKE Class Weight: {proba[1]:.1%}")

                # --- Conditional Word Analysis Section ---
                if selected_model in CLASSICAL_MODELS and X_transformed is not None:
                    st.markdown("---")
                    
                    df_explanation = explain_prediction(vectorizer, model, selected_model, X_transformed)
                    
                    if df_explanation is not None and not df_explanation.empty:
                        if selected_model == "Logistic Regression":
                            if label == "FAKE":
                                st.markdown("### 🚨 Why is this article flagged as **FAKE**?")
                                st.write("The following key tokens inside the text exerted the strongest statistical push towards a **FAKE** classification outcome:")
                                
                                fake_words = df_explanation[df_explanation["Impact"] > 0].sort_values(by="Impact", ascending=False).head(5)
                                
                                if not fake_words.empty:
                                    st.dataframe(fake_words[["Word", "Impact"]], hide_index=True, use_container_width=True)
                                    st.bar_chart(fake_words.set_index("Word")["Impact"], color="#ff4b4b")
                                else:
                                    st.info("No specific strong FAKE indicators found in the vocabulary.")
                                    
                            elif label == "REAL":
                                st.markdown("### ✅ Why is this article flagged as **REAL**?")
                                st.write("The following key tokens inside the text exerted the strongest statistical push towards a **REAL** classification outcome:")
                                
                                real_words = df_explanation[df_explanation["Impact"] < 0].sort_values(by="Impact", ascending=True).head(5)
                                real_words["Abs_Impact"] = real_words["Impact"].abs()
                                
                                if not real_words.empty:
                                    st.dataframe(real_words[["Word", "Impact"]], hide_index=True, use_container_width=True)
                                    st.bar_chart(real_words.set_index("Word")["Abs_Impact"], color="#28a745")
                                else:
                                    st.info("No specific strong REAL indicators found in the vocabulary.")

                        elif selected_model == "Random forest":
                            st.markdown(f"### 🔑 Token Feature Significance Map ({label} Context)")
                            st.write(f"The ensemble forest structure isolated these specific tokens as highly influential decision weights:")
                            
                            top_rf_words = df_explanation.head(10)
                            st.dataframe(top_rf_words[["Word", "Importance"]], hide_index=True, use_container_width=True)
                            st.bar_chart(top_rf_words.set_index("Word")["Importance"], color="#007bff")
                    else:
                        st.info("Insufficient features matched the model's vocabulary array to provide token-level telemetry.")

                # Ground Truth Check for Samples
                if sample_choice != "— Select a sample —":
                    sample = SAMPLES.get(sample_choice)
                    if sample:
                        st.markdown("---")
                        st.info(f"📋 **Evaluation Standard:** Ground Truth Label for this specific validation target is **{sample['label']}**.")



with tab3:
    
    st.title("⚙️ How It Works")

    st.write("""
    The prediction process follows several steps:
    """)

    st.markdown("""
    ### 1️⃣ Input News

    Paste the news article or headline into the Detection page.

    ### 2️⃣ Text Preprocessing

    The text is cleaned by:
    - converting to lowercase
    - removing punctuation
    - removing stop words
    - lemmatization

    ### 3️⃣ Feature Extraction

    The cleaned text is transformed into numerical vectors using
    **TF-IDF Vectorization**.

    ### 4️⃣ Machine Learning Prediction

    The selected Machine Learning model analyzes the feature vector and predicts
    whether the article is **Real** or **Fake**.

    ### 5️⃣ Display Result

    The application displays:
    - Prediction (Real/Fake)
    - Confidence Score
    """)

with tab4:
    st.title("📖 About This Project")

    st.write("""
    The Fake News Detection System is a Machine Learning application developed
    to classify news articles as **Real** or **Fake** based on their textual
    content.

    The project applies Natural Language Processing (NLP) techniques to
    transform news text into numerical features before making predictions with
    trained Machine Learning models.

    This project was developed for educational purposes as part of an
    Artificial Intelligence and Machine Learning course.
    """)

    st.markdown("---")

    st.subheader("📂 Dataset")

    st.write("""
    **Dataset:** REAL and FAKE news dataset.

    The dataset contains labeled news articles collected from multiple online
    sources and includes both real and fake news.

    During this project, the models were primarily trained and evaluated on
    **United States political news**, which means predictions outside this
    domain should be interpreted with caution.
    """)

    st.markdown("---")

    st.subheader("🤖 Machine Learning Models")

    st.write("""
    This application compares two Machine Learning algorithms:

    • Logistic Regression

    • Random Forest

    Both models were trained using TF-IDF text features after preprocessing the
    news articles.
    """)

    st.markdown("---")

    st.subheader("🛠 Technologies")

    st.write("""
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    - NumPy
    - TF-IDF Vectorization
    """)


with tab5:
    st.title("ℹ️ Disclaimer")

    st.warning("""
    This application is designed as an educational Machine Learning project.

    The models were trained primarily using **United States political news**.
    As a result, prediction performance is expected to be strongest for that
    domain.

    News articles related to entertainment, sports, finance, health,
    technology, science, or other topics may not be classified as accurately.

    The prediction should not be considered a substitute for professional
    fact-checking. Users are encouraged to verify important information using
    reputable and trusted news sources.
    """)