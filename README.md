# RNN Sentiment Analysis

Deep learning sentiment classification system built using a Recurrent Neural Network (RNN) architecture. Originally developed as part of an NLP course at UC Santa Cruz and later expanded into a structured portfolio project.

## Overview

This project implements an end-to-end natural language processing pipeline for binary sentiment classification. The system includes text preprocessing, tokenization, embedding layers, model training, hyperparameter tuning, and quantitative evaluation.

The objective was to design and evaluate a neural network capable of capturing sequential dependencies in text data while maintaining modularity and reproducibility.

## Features

- End-to-end NLP pipeline implementation
- Text preprocessing and tokenization workflows
- RNN-based model architecture for sequential text modeling
- Hyperparameter tuning and iterative refinement
- Quantitative evaluation using accuracy, precision, recall, and F1-score

## Tech Stack

- Python
- PyTorch (replace with TensorFlow if applicable)
- NumPy
- Scikit-learn (if used)

## Design Considerations

- Structured training and evaluation loops for reproducibility
- Modular separation of preprocessing, model definition, and evaluation
- Iterative tuning of learning rate, hidden size, and sequence handling
- Performance analysis across multiple metrics to avoid accuracy-only bias

## Future Improvements

- Experimentation with LSTM or GRU architectures
- Incorporating pretrained embeddings (e.g., GloVe)
- Adding model checkpointing and experiment tracking
- Exploring transformer-based architectures for comparison
