import os
import torch
import random
import numpy as np
from glob import glob
# import torch.nn as nn
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models import UNet
from solver import Solver
from models import dc_loss
from dataset import MRIDataset


# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# hyperparameters
LR = 0.001
RESIZE = 32
EPOCHS = 200
PATIENCE = 5
L2_REG = 0.001
BATCH_SIZE = 2
RESUME_TRAIN = True
DATA_PATH = "./data/kaggle_3m/*/*"

# main script
if __name__ == "__main__":
    # shuffle and split dataset into training and testing sets
    all_images = shuffle([fn for fn in glob(pathname=DATA_PATH, recursive=True) if "_mask" not in fn], random_state=SEED)
    X_train, X_test = train_test_split(all_images, test_size=0.2, random_state=SEED)

    # create datasets and apply data augmentation
    train_dataset = MRIDataset(images=X_train, resize_h=RESIZE, resize_w=RESIZE, data_aug=True)
    test_dataset = MRIDataset(images=X_test, resize_h=RESIZE, resize_w=RESIZE, data_aug=False)

    # dataLoader settings for efficient data loading
    pin = True if torch.cuda.is_available() else False

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=4, pin_memory=pin) # os.cpu_count()
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=pin) # os.cpu_count()
        
    # determine the device for training (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # instantiate the U-Net model and move it to the specified device
    model = UNet().to(device)

    if RESUME_TRAIN:
        print("\nResuming training...")
        model.load_state_dict(torch.load(f="./checkpoints/model-epoch=1-val_loss=0.5930.pth", 
                                         map_location=device))

    # define the optimizer for training the model
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, 
                                 betas=(0.9, 0.999), weight_decay=L2_REG)
    
    # define the loss function for training the model
    loss_fn = dc_loss # loss_fn = nn.BCEWithLogitsLoss()

    # create an instance of the Solver class for training and validation
    solver = Solver(epochs=EPOCHS,
                    writer=None,
                    train_loader=train_loader, 
                    test_loader=test_loader, 
                    device=device, 
                    model=model, 
                    optimizer=optimizer, 
                    criterion=loss_fn, 
                    patience=PATIENCE)

    # train the neural network
    solver.train_net()

    # check the model ability
    solver.check_results()
