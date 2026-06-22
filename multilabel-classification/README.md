# 🧠 Multilabel Classification in PyTorch

This module provides a **PyTorch implementation of a multilabel classification pipeline**, designed for computer vision tasks where each input sample can belong to multiple classes simultaneously.

The focus is on a **clean, modular and reusable code structure** for training and evaluating multilabel models.

## 🎯 Objective

The goal of this implementation is to provide a complete workflow for multilabel classification, including:

- dataset preparation with multi-hot encoded labels
- model definition for multilabel outputs
- training pipeline using appropriate loss functions
- evaluation and metric computation
- inference on unseen data

The code is designed to be **adaptable to different multilabel vision problems and datasets**.

## 🧩 What is Multilabel Classification?

In multilabel classification, each input sample can be associated with **multiple labels at the same time**.

Unlike single-label classification:
- outputs are not mutually exclusive
- multiple classes can be active simultaneously
- predictions are typically modeled using independent probabilities per class

## ⚙️ Pipeline Overview

The typical workflow implemented in this module is:

1. Load dataset with multilabel annotations
2. Convert labels into multi-hot vectors
3. Apply preprocessing and augmentations
4. Define a neural network with sigmoid outputs
5. Train using multilabel loss functions
6. Evaluate model performance using multilabel metrics
7. Run inference on new samples
8. Apply thresholding for final predictions

## 🧠 Model Approach

The implementation follows a standard multilabel setup:

- Output layer size = number of classes
- No softmax activation (independent class probabilities)
- Sigmoid activation applied per class
- Loss function: **BCEWithLogitsLoss**

This formulation allows each class to be learned as an independent binary classification problem.


## ⚙️ Tech Stack

- PyTorch
- Torchvision
- NumPy
- OpenCV
- Scikit-learn (metrics)

## 🧪 Key Features

- End-to-end multilabel classification pipeline
- Support for custom datasets
- BCEWithLogitsLoss-based training
- Flexible thresholding for inference
- Modular PyTorch codebase
- Evaluation using multilabel metrics (F1-score, precision/recall)

## 📊 What this module demonstrates

This implementation demonstrates the ability to:

- design and implement multilabel classification systems in PyTorch
- correctly handle **multi-hot label representations**
- build robust training and evaluation pipelines
- apply proper loss functions for multilabel learning
- structure scalable computer vision codebases
