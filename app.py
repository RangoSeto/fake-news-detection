"""Streamlit demo for fake news classification (CST-3121)."""

from typing import Any

from pathlib import Path

import joblib
import streamlit as st

from preprocessing import build_full_text
from samples import SAMPLES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "AI_Project" / "Output" 
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
  return pred, proba


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


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("Fake News Detector")
st.caption("CST-3121 Group Project — Baseline ML + DistilBERT Demo")

vectorizer, classical_models = load_classical_artifacts()
model_options = available_models()

with st.sidebar:
  st.header("About")
  st.write(
    "Classifies news articles as **REAL** or **FAKE** using TF-IDF models "
    "or a fine-tuned DistilBERT transformer."
  )
  if BERT_AVAILABLE:
    st.write("**Models:** Logistic Regression, Linear SVM, DistilBERT")
  else:
    st.write("**Models:** Logistic Regression, Linear SVM (DistilBERT unavailable locally)")

  if BERT_AVAILABLE:
    st.write("**Best model (test):** DistilBERT (~96.8% accuracy)")
  else:
    st.write("**Best model (test):** Logistic Regression (~93.2% accuracy)")

  st.header("How it works")
  st.markdown(
    "1. Clean title and article text\n"
    "2. **TF-IDF models:** convert to sparse features, classify with sklearn\n"
    "3. **DistilBERT:** tokenize text, classify with transformer embeddings"
  )

  if BERT_AVAILABLE:
    st.info(
      "DistilBERT loads on first use (~257 MB). First prediction may take "
      "10–30 seconds on CPU."
    )
  else:
    st.info(
      "Run `notebooks/04_distilbert_finetuning.ipynb` so "
      "`outputs/models/distilbert-best/` exists locally."
    )

  st.header("Limitations")
  st.markdown(
    "- Dataset is US-politics-heavy; may not generalize\n"
    "- False positives can flag legitimate news\n"
    "- Patterns may reflect source bias, not factual falsity"
  )

  st.warning(
    "Academic demo only. Do not use as a sole fact-checker. "
    "Results can be wrong."
  )

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
  f"{selected_model}: {metrics['accuracy']:.1%} accuracy "
  f"({metrics['split']} set)"
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

title = st.text_input("Article title", key="title_input")
body = st.text_area("Article text", height=200, key="body_input")

if body.strip():
  st.caption(f"{len(body.split())} words · {len(body)} characters")

analyze = st.button("Analyze", type="primary")

if analyze:
  if not title.strip() and not body.strip():
    st.error("Please enter a title or article text.")
  else:
    if len(body.strip()) < 20:
      st.warning("Article text is very short. Results may be unreliable.")

    full_text = build_full_text(title, body)

    if not full_text:
      st.error("No valid text after cleaning.")
    else:
      spinner_text = (
        "Loading DistilBERT and analyzing..."
        if selected_model == DISTILBERT_LABEL
        else "Analyzing..."
      )
      with st.spinner(spinner_text):
        if selected_model == DISTILBERT_LABEL:
          bert_pipe = load_bert_pipeline()
          pred, proba = predict_bert(bert_pipe, full_text)
        else:
          model = classical_models[selected_model]
          pred, proba = predict_classical(vectorizer, model, full_text)

      label = LABEL_MAP[pred]
      confidence = proba[pred]

      if label == "REAL":
        st.success(f"Prediction: **{label}**")
      else:
        st.error(f"Prediction: **{label}**")

      st.metric("Confidence", f"{confidence:.1%}")

      with st.expander("Probability breakdown"):
        st.write(f"REAL: {proba[0]:.1%}")
        st.write(f"FAKE: {proba[1]:.1%}")

      if sample_choice != "— Select a sample —":
        sample = SAMPLES.get(sample_choice)
        if sample:
          st.info(f"Known label for this sample: **{sample['label']}**")
