# News Classification (Production-style Example)

This repository contains a **production-oriented example** of a news classification pipeline:
from data extraction and preprocessing to model training and inference.

> Note: Some parts may be incomplete or simplified.

---

## Project Goals

- Build an end-to-end pipeline for **news classification** (e.g., political pressure / institutional risk topics, etc.)
- Provide two modeling approaches:
  1) **Classical ML baseline** (linear models)
  2) **LLM-based classification** (OpenAI API) for fast iteration / weak supervision / labeling

---

## Repository Structure

### `data_extractor/`
Responsible for:
- fetching raw news data (API / files / dumps)
- converting sources into a unified raw format (JSON/CSV)
- saving intermediate artifacts

### `data_preprocessing/`
Responsible for:
- text cleaning (deduplication, normalization)
- language filtering (optional)
- label mapping / dataset formatting
- train/valid/test splitting

### `linear_model/`
A reproducible baseline:
- TF-IDF / bag-of-words features
- Logistic Regression / Linear SVM (optional)
- evaluation and reporting

### `openai_model/`
LLM-based classifier:
- prompt templates for consistent labeling
- batch inference / cost-aware processing
- saving predictions + rationales (optional)




