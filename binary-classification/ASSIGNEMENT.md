# Street Cleaning Binary Classification

A street cleaning company aims to ensure the effectiveness of their cleaning efforts by verifying whether the streets are clean or contain garbage. This repository contains a solution for a binary classification problem using computer vision. The dataset consists of images representing clean streets (class 0: "clean") and streets with garbage (class 1: "dirty"). The goal is to develop a machine learning model that can accurately distinguish between clean and dirty streets.

## Dataset
The dataset is available on Kaggle at [CleanDirty Road Classification](https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification/). It includes a total of 237 images sourced from the internet. The images are divided into two classes: "clean" and "dirty." The dataset also provides a CSV file, `metadata.csv`, mapping each image's filename to its corresponding class label.

### Dataset Structure
- **Images:** Folder containing all road images.
- **metadata.csv:** CSV file mapping image names to class labels.

<p align="center">
    <img src="./imgs/dataset.png" alt="Dataset examples">
</p>

## Project Structure

- **main.py:** Script for running the training and validation of the model in k-fold cross-validation or hold-out split validation mode.
- **custom_dataset.py:** Custom dataset class for loading and preprocessing images.
- **early_stopping.py:** Implementation of an early stopping mechanism to prevent overfitting.
- **hold_out_split_validation.py:** Script to launch training and hold-out split validation for model evaluation.
- **hyperparameters_optimization.py:** Script for hyperparameter optimization (optimizer, learning rate, etc.) using Optuna.
- **inference.py:** Inference script for making predictions on new test data.
- **k_fold_cross_validation.py:** Script to launch training and k-fold cross-validation for robust model evaluation.
- **plotting_utils.py:** Utility functions for plotting results.
- **solver.py:** Training and validation logic.
- **stats.py:** Computation of statistics for training and validation and plotting.
- **keras-binary-classification/main.py:** Script for running the training and validation of the model in k-fold cross-validation with keras-tensorflow.

## Installation

1. Clone the repository:

    ```bash
    https://github.com/iamvincenzo/pytorch-deep-learning-computer-vision.git
    cd binary-classification
    ```

2. Create a Conda environment:

    ```bash
    conda create -n street_classification
    conda activate street_classification
    ```

2. Install dependencies:

    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    conda install -c anaconda pillow
    conda install -c conda-forge matplotlib
    conda install -c conda-forge pytorch-model-summary
    conda install -c conda-forge torchmetrics
    conda install -c conda-forge optuna
    conda install -c anaconda scikit-learn
    conda install -c anaconda numpy
    conda install -c anaconda pandas
    conda install -c anaconda tqdm
    ```

## How to Run

1. Navigate to the project directory:

    ```bash
    cd binary-classification
    ```

2. Run the main program:

    ```bash
    python main.py
    ```

    ### Command-line Arguments

    The following command-line arguments can be used when running the script:

    - `--run_name`: The name assigned to the current run.
    - `--model_name`: The name of the model to be saved or loaded.
    - `--num_epochs`: The total number of training epochs.
    - `--batch_size`: The batch size for training and validation data.
    - `--workers`: The number of workers in the data loader.
    - `--lr`: The learning rate for optimization.
    - `--loss`: The loss function used for model optimization.
    - `--opt`: The optimizer used for training.
    - `--patience`: The threshold for early stopping during training.
    - `--load_model`: Determines whether to load the model from a checkpoint.
    - `--checkpoint_path`: The path to save the trained model.
    - `--num_classes`: The number of classes to predict with the final Linear layer.
    - `--raw_data_path`: Path where to get the raw dataset.
    - `--apply_transformations`: Indicates whether to apply transformations to images.

## Results

<p align="center">
    <img src="./imgs/clean.png" alt="Clean Street prediction">
</p>

<p align="center">
    <img src="./imgs/dirty.png" alt="Dirty Street prediction">
</p>

## License

This project is licensed under the [GNU GENERAL PUBLIC LICENSE  Version 3](../LICENSE).
