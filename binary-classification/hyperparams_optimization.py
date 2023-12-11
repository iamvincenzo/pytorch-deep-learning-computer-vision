"""
Credits: https://github.com/optuna/optuna-examples/tree/main/pytorch
"""
import os
import torch
import optuna
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from optuna.trial import TrialState
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split

from custom_dataset import CleanDirtyRoadDataset

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# parameters
RESIZE = 224
PATIENCE = 5
NUM_CLASSES = 1
BATCH_SIZE = 16
NUM_EPOCHS = 200
DIR = os.getcwd()
APPLY_TRANSF = False
CRITERION = nn.BCEWithLogitsLoss()
N_TRAIN_EXAMPLES = BATCH_SIZE * 30
N_VALID_EXAMPLES = BATCH_SIZE * 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_img_pth = Path("./data/Images")
metadata_path = Path("./data/metadata.csv")


def finetuning():
    """
    Create a ResNet18 model with pre-trained weights and customize the final layer for fine-tuning.

    Returns:
        torch.nn.Module: ResNet18 model with a modified final layer.
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # add a new final layer
    nr_filters = model.fc.in_features
    model.fc = nn.Linear(nr_filters, NUM_CLASSES)

    return model


def define_model(trial):
    """
    Define a model for Optuna hyperparameter optimization.

    Args:
        trial (optuna.Trial): The trial object for hyperparameter optimization.

    Returns:
        torch.nn.Module: A PyTorch model with hyperparameter-specific architecture.
    """
    pass


def get_loaders():
    """
    Prepare DataLoader instances for training and testing.

    Returns:
        DataLoader: Training DataLoader.
        DataLoader: Testing DataLoader.
    """
    df = pd.read_csv(filepath_or_buffer=metadata_path)

    X_train, X_test, y_train, y_test = train_test_split(df["filename"], df["label"],
                                                        test_size=0.2, random_state=SEED, shuffle=True)

    # define data transformations for training and testing
    if APPLY_TRANSF:
        train_transform = transforms.Compose([transforms.Resize(size=(RESIZE, RESIZE)),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5),
                                              transforms.RandomRotation(degrees=15),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize(size=(RESIZE, RESIZE)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
    else:
        train_transform = None
        test_transform = None

    # create instances of the custom dataset for training and testing
    train_dataset = CleanDirtyRoadDataset(X=X_train, y=y_train, resize=RESIZE,
                                          data_root=base_img_pth, transform=train_transform)
    test_dataset = CleanDirtyRoadDataset(X=X_test, y=y_test, resize=RESIZE,
                                         data_root=base_img_pth, transform=test_transform)

    # pin_memory: speed up the host (cpu) to device (gpu) transfer
    pin = True if torch.cuda.is_available() else False

    # create DataLoader instances for training and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=os.cpu_count(), pin_memory=pin)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=os.cpu_count(), pin_memory=pin)

    return train_loader, test_loader


def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.Trial): The trial object for hyperparameter optimization.

    Returns:
        float: Accuracy obtained with the hyperparameter configuration.
    """
    # generate the model
    model = finetuning().to(DEVICE)

    # generate the optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # get the dataset
    train_loader, test_loader = get_loaders()

    # training of the model
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # # Limiting training data for faster epochs.
            # if batch_idx * BATCH_SIZE >= N_TRAIN_EXAMPLES:
            #     break
            data = data.to(DEVICE)
            target = target.unsqueeze(1).to(DEVICE)
            logits = model(data)
            loss = CRITERION(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation of the model
        predictions = torch.tensor([], device=DEVICE)
        targets = torch.tensor([], device=DEVICE)
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, y_valid) in enumerate(test_loader):
                # # Limiting validation data.
                # if batch_idx * BATCH_SIZE >= N_VALID_EXAMPLES:
                #     break                
                data = data.to(DEVICE)
                y_valid = y_valid.unsqueeze(1).to(DEVICE)                
                logits = model(data)                                
                # since we are using BCEWithLogitsLoss
                # logits --> probabilities --> labels
                probs = torch.sigmoid(logits)
                y_pred = torch.round(probs)
                # record predictions and true labels
                predictions = torch.cat([predictions, y_pred], dim=0)
                targets = torch.cat([targets, y_valid], dim=0)
            targets = targets.squeeze()
            predictions = predictions.squeeze()
            accuracy = torch.sum(predictions == targets).item() / targets.size(0)

        trial.report(accuracy, epoch)

        # handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
