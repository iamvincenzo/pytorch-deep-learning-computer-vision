import os
import cv2
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
RESIZE = 224
PATIENCE = 5
BATCH_SIZE = 16

TEST_IMGS_PATH = Path("./data/TEST")
TRAIN_IMGS_PATH = Path("./data/TRAIN")
METADATA_TEST_PATH = Path("./data/test_labels.csv")
METADATA_TRAIN_PATH = Path("./data/train_labels.csv")

FIXED_TRAIN_FILE_PATH = "./data/train_labels_fixed.csv"
FIXED_TEST_FILE_PATH = "./data/test_labels_fixed.csv"

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def clean_dataframe(df, imgs_root_path, save_path):
    """
    Offline dataframe preprocessing.
    """
    # remove images that don't exist from training data
    df = df[[filename in os.listdir(imgs_root_path) for filename in df["filename"]]]
    
    # fix items: there are some values of width (w) and height (h) 
    # that are either set to 0 or do not match those of the actual images
    for idx, row in df.iterrows():
        filename = row["filename"]
        img = cv2.imread(str(imgs_root_path / filename))
    
        if img is not None:
            height, width, _ = img.shape
            if row["height"] != height or row["width"] != width:
                df.at[idx, "height"] = height
                df.at[idx, "width"] = width

    condition = df["width"] < df["xmax"]
    df = df.drop(df[condition].index)

    condition = df["height"] < df["ymax"]
    df = df.drop(df[condition].index)

    # compute bbox are for min area value to pass to albumentations
    df["area"] = (df["xmax"] - df["xmin"]) * (df["ymax"] - df["ymin"])

    # export the DataFrame to a CSV file
    df.to_csv(save_path, index=False)

    return df


if __name__ == "__main__":
    if not os.path.exists(FIXED_TRAIN_FILE_PATH):
        # read train metadata
        train_df = pd.read_csv(filepath_or_buffer=METADATA_TRAIN_PATH)
        # clean train dataframe
        train_df = clean_dataframe(df=train_df, imgs_root_path=TRAIN_IMGS_PATH, save_path=FIXED_TRAIN_FILE_PATH)
    else:
        # read train metadata
        train_df = pd.read_csv(filepath_or_buffer=FIXED_TRAIN_FILE_PATH)

    if not os.path.exists(FIXED_TEST_FILE_PATH):
        # read test metadata
        test_df = pd.read_csv(filepath_or_buffer=METADATA_TEST_PATH)
        # clean test dataframe
        test_df = clean_dataframe(df=test_df, imgs_root_path=TEST_IMGS_PATH,  save_path=FIXED_TEST_FILE_PATH)
    else:
        # read train metadata
        test_df = pd.read_csv(filepath_or_buffer=FIXED_TEST_FILE_PATH)
    
    # prepare class labels and mappings
    # reload not cleaned train_df because it can contain classes that are in test_df but not in cleaned train_df 
    train_df_classes_original = pd.read_csv(filepath_or_buffer=METADATA_TRAIN_PATH)
    classes_list = sorted(train_df_classes_original["class"].unique())
    classes_list.insert(0, "__background__")
    train_classes = {cls: idx for idx, cls in enumerate(classes_list, start=0)}
    train_classes_inverse = {idx: cls for idx, cls in enumerate(classes_list, start=0)}
    NUM_CLASSES = len(classes_list)

    # determine computing device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create training and test datasets
    train_dataset = PlantDocDataset(df=train_df, data_aug=True, resize_h=RESIZE, resize_w=RESIZE,
                                    root=TRAIN_IMGS_PATH, classes=train_classes, classes_inverse=train_classes_inverse)    
    test_dataset = PlantDocDataset(df=test_df, data_aug=False, resize_h=RESIZE, resize_w=RESIZE,
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
