# Street Cleaning Binary Classification

[![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-orange?style=flat-square&logo=ubuntu&logoColor=white)](https://ubuntu.com/) [![Windows](https://img.shields.io/badge/Windows-11-blue?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/windows/) [![VS Code](https://img.shields.io/badge/VS%20Code-v1.61.0-007ACC?style=flat-square&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/) [![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-v1.10.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Keras](https://img.shields.io/badge/Keras-2.8.0-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.8.0-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/) [![Optuna](https://img.shields.io/badge/Optuna-v2.10.0-EE4D5A?style=flat-square&logo=optuna&logoColor=white)](https://optuna.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.4.3-FF5733?style=flat-square&logo=python&logoColor=white)](https://matplotlib.org/) [![NumPy](https://img.shields.io/badge/NumPy-v1.21.0-4C65AF?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/) [![Pandas](https://img.shields.io/badge/Pandas-v1.3.3-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

A street cleaning company aims to ensure the effectiveness of their cleaning efforts by verifying whether the streets are clean or contain garbage. This repository contains a solution for a binary classification problem using computer vision. The dataset consists of images representing clean streets (class 0: "clean") and streets with garbage (class 1: "dirty"). The goal is to develop a machine learning model that can accurately distinguish between clean and dirty streets.

## Dataset
The dataset is available on Kaggle at [CleanDirty Road Classification](https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification/). It includes a total of 237 images sourced from the internet. The images are divided into two classes: "clean" and "dirty." The dataset also provides a CSV file, `metadata.csv`, mapping each image's filename to its corresponding class label.

<p align="center">
    <img src="./imgs/dataset.png" alt="Dataset examples">
</p>

## Project Structure

- **main.py:** Script for running the training and validation of the model in k-fold cross-validation or hold-out split validation mode.
- **custom_dataset.py:** Custom dataset class for loading and preprocessing images.
- **early_stopping.py:** Implementation of an early stopping mechanism to prevent overfitting.
- **hyperparameters_optimization.py:** Script for hyperparameter optimization (optimizer, learning rate, etc.) using Optuna.
- **hold_out_split_validation.py:** Script to launch training and hold-out split validation for model evaluation.
- **k_fold_cross_validation.py:** Script to launch training and k-fold cross-validation for robust model evaluation.
- **solver.py:** Training and validation logic.
- **plotting_utils.py:** Utility functions for plotting results.
- **stats.py:** Computation of statistics for training and validation and plotting.
- **inference.py:** Inference script for making predictions on new test data.

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

<table>
<tr>
<th> Train configuration </th>
<th> Test results </th>
</tr>
<tr>
<td>

| Parameter                 | Value          |
|---------------------------|----------------|
| Model                | ResNet18             |
| Batch Size                | 16             |
| Learning Rate             | 0.004014796    |
| Optimizer                 | RMSprop        |
| Transformations           | Flip, Rotations, Normalization |

</td>

<td>

| Metric                    | Average        |
|---------------------------|----------------|
| Accuracy                  | 92.846%        |
| Loss                      | 0.169          |

</td>
</tr> 
</table>

<table>
<tr>
<th> Train configuration </th>
<th> Test results </th>
</tr>
<tr>
<td>

| Parameter                 | Value          |
|---------------------------|----------------|
| Model                | ResNet18             |
| Batch Size                | 16             |
| Learning Rate             | 0.004014796    |
| Optimizer                 | RMSprop        |
| Transformations           | False          |

</td>

<td>

| Metric                    | Average        |
|---------------------------|----------------|
| Accuracy                  | 95.780%        |
| Loss                      | 0.107          |

</td>
</tr> 
</table>

</td>
</tr> 
</table>

<table>
<tr>
<th> Train configuration </th>
<th> Test results </th>
</tr>
<tr>
<td>

| Parameter                 | Value          |
|---------------------------|----------------|
| Model                | ResNet18             |
| Batch Size                | 16             |
| Learning Rate             | 0.004014796    |
| Optimizer                 | RMSprop        |
| Apply Transformations     | TrivialAugmentWide           |

</td>

<td>

| Metric                    | Average        |
|---------------------------|----------------|
| Accuracy                  | 95.798%        |
| Loss                      | 0.103          |

</td>
</tr> 
</table>

<table>
<tr>
<th> Train configuration </th>
<th> Test results </th>
</tr>
<tr>
<td>

| Parameter                 | Value          |
|---------------------------|----------------|
| Model                | ResNet18             |
| Batch Size                | 16             |
| Learning Rate             | 0.001          |
| Optimizer                 | Adam           |
| Transformations           | Flip, Rotations, Normalization |

</td>

<td>

| Metric                    | Average        |
|---------------------------|----------------|
| Accuracy                  | 95.771%        |
| Loss                      | 0.120          |

</td>
</tr> 
</table>

<table>
<tr>
<th> Train configuration </th>
<th> Test results </th>
</tr>
<tr>
<td>

| Parameter                 | Value          |
|---------------------------|----------------|
| Model                | ResNet18             |
| Batch Size                | 16             |
| Learning Rate             | 0.001          |
| Optimizer                 | Adam           |
| Apply Transformations    | TrivialAugmentWide           |

</td>

<td>

| Metric                    | Average        |
|---------------------------|----------------|
| Accuracy                  | 96.622%        |
| Loss                      | 0.095          |

</td>
</tr> 
</table>

<table>
<tr>
<th> Train configuration </th>
<th> Test results </th>
</tr>
<tr>
<td>

| Parameter                 | Value          |
|---------------------------|----------------|
| Model                | ResNet18             |
| Batch Size                | 16             |
| Learning Rate             | 0.001          |
| Optimizer                 | Adam           |
| Apply Transformations    | False          |

</td>

<td>

| Metric                    | Average        |
|---------------------------|----------------|
| Accuracy                  | 96.197%        |
| Loss                      | 0.091          |

</td>
</tr> 
</table>

<table>
<tr>
<th> Train configuration </th>
<th> Test results </th>
</tr>
<tr>
<td>

| Parameter                 | Value          |
|---------------------------|----------------|
| Model                | MobileNetV2             |
| Batch Size                | 16             |
| Learning Rate             | 0.001          |
| Optimizer                 | Adam           |
| Apply Transformations    | TrivialAugmentWide           |

</td>

<td>

| Metric                    | Average        |
|---------------------------|----------------|
| Accuracy                  | 96.622%        |
| Loss                      | 0.098          |

</td>
</tr> 
</table>

<table>
<tr>
<th> Train configuration </th>
<th> Test results </th>
</tr>
<tr>
<td>

| Parameter                 | Value          |
|---------------------------|----------------|
| Model                | EfficientNet-V2-L            |
| Batch Size                | 16             |
| Learning Rate             | 0.001          |
| Optimizer                 | Adam           |
| Apply Transformations    | TrivialAugmentWide           |

</td>

<td>

| Metric                    | Average        |
|---------------------------|----------------|
| Accuracy                  | 96.613%        |
| Loss                      | 0.171          |

</td>
</tr> 
</table>

<table>
<tr>
<th> Train configuration </th>
<th> Test results </th>
</tr>
<tr>
<td>

| Parameter                 | Value          |
|---------------------------|----------------|
| Model                | VGG-16          |
| Batch Size                | 16             |
| Learning Rate             | 0.001          |
| Optimizer                 | Adam           |
| Apply Transformations    | TrivialAugmentWide           |

</td>

<td>

| Metric                    | Average        |
|---------------------------|----------------|
| Accuracy                  | 96.613%        |
| Loss                      | 0.103          |

</td>
</tr> 
</table>

<p align="center">
    <img src="./imgs/clean.png" alt="Clean Street prediction">
</p>

<p align="center">
    <img src="./imgs/dirty.png" alt="Dirty Street prediction">
</p>

## License

This project is licensed under the [GNU GENERAL PUBLIC LICENSE  Version 3](../LICENSE).
