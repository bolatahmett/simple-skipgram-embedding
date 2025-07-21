# Mini Skip-Gram Model in Python

This repository contains a simple implementation of the Skip-Gram word embedding model from scratch using only NumPy.  
It trains embeddings on a small custom dataset and predicts context words given a target word.

## Features

- Builds vocabulary and mappings
- Prepares training pairs from sample sentences
- Trains embeddings with Skip-Gram and softmax output
- Uses cross-entropy loss and basic gradient descent
- Predicts likely context words for a given input word

## Usage

- Run the training script to train embeddings on example sentences.
- After training, use `predict_context_words(word)` to see probable context words.

## How it works

Skip-Gram trains word embeddings by predicting the context words surrounding a target word in a sentence.  
Embeddings capture semantic relationships by learning which words tend to appear near each other.

## Requirements

- Python 3.x  
- NumPy

