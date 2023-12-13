import os
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from solver import Solver
from dataset import collate_fn
from models import create_model
from dataset import PlantDocDataset


# parameters
SEED = 42
EPOCHS = 200
RESIZE = 128
PATIENCE = 5
BATCH_SIZE = 32

TEST_IMGS_PATH = Path("./data/TEST")
TRAIN_IMGS_PATH = Path("./data/TRAIN")
METADATA_TEST_PATH = Path("./data/test_labels.csv")
METADATA_TRAIN_PATH = Path("./data/train_labels.csv")

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


if __name__ == "__main__":
    # read and filter training metadata (remove images that don't exist)
    train_df = pd.read_csv(filepath_or_buffer=METADATA_TRAIN_PATH)
    filtered_train_df = train_df[[os.path.isfile(TRAIN_IMGS_PATH / filename) for filename in train_df["filename"]]]

    # read and filter testing metadata (remove images that don't exist)
    test_df = pd.read_csv(filepath_or_buffer=METADATA_TEST_PATH)
    filtered_test_df = test_df[[os.path.isfile(TEST_IMGS_PATH / filename) for filename in test_df["filename"]]]

    # prepare class labels and mappings
    classes_list = sorted(filtered_train_df["class"].unique())
    classes_list.insert(0, "__background__")
    train_classes = {cls: idx for idx, cls in enumerate(classes_list, start=0)}
    train_classes_inverse = {idx: cls for idx, cls in enumerate(classes_list, start=0)}
    NUM_CLASSES = len(classes_list)

    # determine computing device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create training and test datasets
    train_dataset = PlantDocDataset(df=filtered_train_df, transform=None, resize=224, 
                                    root=TRAIN_IMGS_PATH, classes=train_classes, classes_inverse=train_classes_inverse)    
    test_dataset = PlantDocDataset(df=filtered_test_df, transform=None, resize=224, 
                                   root=TEST_IMGS_PATH, classes=train_classes, classes_inverse=train_classes_inverse)
    
    # determine if GPU pinning is possible
    pin = True if torch.cuda.is_available() else False
    
    # create data loaders for training and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=os.cpu_count(), pin_memory=pin, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=os.cpu_count(), pin_memory=pin, collate_fn=collate_fn)

    # initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005) 

    # create the Solver instance
    solver = Solver(epochs=EPOCHS, 
                    device=device, 
                    train_loader=train_loader, 
                    valid_loader=test_loader, 
                    model=model, 
                    optimizer=optimizer,
                    patience=PATIENCE)
    
    # train the neural network
    solver.train_net()
