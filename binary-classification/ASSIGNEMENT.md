# Street Cleaning Binary Classification

## Problem Statement
A street cleaning company aims to ensure the effectiveness of their cleaning efforts by verifying whether the streets are clean or contain garbage. This repository contains a solution for a binary classification problem using computer vision. The dataset consists of images representing clean streets (class 0: "clean") and streets with garbage (class 1: "dirty"). The goal is to develop a machine learning model that can accurately distinguish between clean and dirty streets.

## Dataset
The dataset is available on Kaggle at [CleanDirty Road Classification](https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification/). It includes a total of 237 images sourced from the internet. The images are divided into two classes: "clean" and "dirty." The dataset also provides a CSV file, `metadata.csv`, mapping each image's filename to its corresponding class label.

### Dataset Structure
- **Images:** Folder containing all road images.
- **metadata.csv:** CSV file mapping image names to class labels.

## Project Overview

### 1. Load and Preprocess Data
- Utilize the images provided in the dataset folder.
- Load and preprocess the images for model training.

### 2. Choose an Appropriate Model
- Select a suitable machine learning model for image classification.
- Consider using pre-trained models and data augmentation due to the limited dataset.

### 3. Train the Model
- Implement the chosen model and train it on the preprocessed dataset.
- Fine-tune the model parameters to achieve optimal performance.

### 4. Evaluate Model Performance
- Evaluate the trained model using appropriate metrics.
- Analyze the results and assess the model's ability to classify clean and dirty streets accurately.

## Potential Applications
A successful classification model can be employed in real-world scenarios, such as developing applications to detect littered areas on roads using cameras. This technology could facilitate timely and targeted cleaning efforts, optimizing the resources of the street cleaning company.

<!-- Problema Computer Vision – classificazione binaria

Un’azienda che pulisce strade deve verificare se la pulizia è stata fatta in modo corretto. Si ha a disposizione un dataset di immagini contenenti foto di strade pulite (classe 0: “clean”) e con spazzatura (classe 1: “dirty”). Sviluppare un modello di machine learning di classificazione che distingua se un’immagine è “clean” o “dirty”.

Il dataset è disponibile a questo link:
https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification/

1. Caricare e pre-processare le immagini del dataset.
2. Scegliere il modello di machine learning appropriato.
3. Addestrare il modello.
4. Valutare le prestazioni del modello. -->