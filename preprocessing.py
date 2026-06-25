"""Text preprocessing matching notebooks/01_data_preprocessing.ipynb."""

import re


def clean_text(text: str) -> str:
  if text is None:
    return ""
  if isinstance(text, float) and text != text:  # NaN check without pandas
    return ""
  text = str(text)
  text = re.sub(r"http\S+|www\.\S+", " ", text)
  text = re.sub(r"<[^>]+>", " ", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text


def build_full_text(title: str, body: str) -> str:
  title_clean = clean_text(title)
  text_clean = clean_text(body)
  return f"{title_clean} {text_clean}".strip()
