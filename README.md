# BERT-Based Sentiment Analysis (Multi-Class)

A deep learning sentiment classification system built using a Transformer-based architecture (BERT) in PyTorch. Originally developed as part of an NLP course at UC Santa Cruz and later expanded into a structured portfolio project focused on model comparison, reproducibility, and evaluation rigor.

---

## Overview

This project implements an end-to-end natural language processing pipeline for multi-class sentiment classification using the Yelp Review dataset (5-star labels).

The system includes:

- Data loading via Hugging Face `datasets`
- Tokenization using a pretrained BERT tokenizer
- Fine-tuning a pretrained `bert-base-uncased` model
- Training and validation loops in PyTorch
- Quantitative evaluation across multiple performance metrics
- Baseline comparison using TF-IDF + Logistic Regression

The primary objective was to evaluate modern Transformer architectures against traditional NLP approaches and understand performance trade-offs in text classification.

---

## Dataset

- Yelp Review Full (via Hugging Face Datasets)
- 5-class sentiment labels (1–5 stars)
- Large-scale text classification benchmark

---

## Architecture

### Transformer Model (Primary Model)

- Pretrained model: `bert-base-uncased`
- Fine-tuned for 5-class sentiment classification
- Classification head added on top of BERT encoder
- Loss function: `CrossEntropyLoss`
- Optimizer: `AdamW`

### Classical Baseline

- TF-IDF vectorization
- Logistic Regression classifier
- Used for performance comparison

---

## Features

- End-to-end NLP training pipeline
- Transformer-based fine-tuning
- Baseline model comparison
- Structured PyTorch training loop
- Multi-metric evaluation:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1-score (weighted)
- Inference pipeline for custom text input

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- NumPy

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
torch
transformers
datasets
scikit-learn
sentencepiece
sacremoses
numpy
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Launch Jupyter Notebook:

```bash
jupyter notebook
```

3. Run the notebook cells sequentially to:
   - Load and preprocess the dataset
   - Train the BERT classifier
   - Evaluate model performance
   - Train the TF-IDF + Logistic Regression baseline
   - Compare results

---

## Design Considerations

- Clear separation between preprocessing, model definition, and evaluation
- Reproducible training loop structure
- Multi-metric evaluation to avoid accuracy-only bias
- Baseline comparison to contextualize Transformer performance

---

## Future Improvements

- Add LSTM/GRU baseline for sequential modeling comparison
- Tokenize dataset using `datasets.map()` for improved efficiency
- Implement model checkpointing and early stopping
- Move training into standalone `train.py` script
- Add experiment tracking (e.g., Weights & Biases)

---

## License

MIT License
