import os
import cv2
import torch
import random
import numpy as np
from glob import glob
import torch.nn as nn
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import UNet
from solver import Solver
from models import dc_loss
from dataset import MRIDataset
from early_stopping import load_checkpoint


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
PATIENCE = 10
start_epoch = 0
L2_REG = 0 # 0.004
BATCH_SIZE = 32
FINETUNING = False
RESUME_TRAIN = False
TRAIN_ONLY_TUMOR = True
DATA_PATH = "./data/kaggle_3m/*/*"
WORKERS = os.cpu_count() if os.cpu_count() < 4 else 4


# main script
if __name__ == "__main__":
    # shuffle and split dataset into training and testing sets
    all_images = shuffle([fn for fn in glob(pathname=DATA_PATH, recursive=True) if "_mask" not in fn], random_state=SEED)
    
    if TRAIN_ONLY_TUMOR:
        all_masks = [fn for fn in glob(pathname=DATA_PATH, recursive=True) if "_mask" in fn]
        tumor = [fn for fn in all_masks if np.any(cv2.imread(fn, cv2.IMREAD_GRAYSCALE))]
        all_images = [fn.replace("_mask", "") for fn in tumor]

    X_train, X_test = train_test_split(all_images, test_size=0.2, random_state=SEED)

    # create datasets and apply data augmentation
    train_dataset = MRIDataset(images=X_train, resize_h=RESIZE, resize_w=RESIZE, data_aug=True)
    test_dataset = MRIDataset(images=X_test, resize_h=RESIZE, resize_w=RESIZE, data_aug=False)

    # dataLoader settings for efficient data loading
    pin = True if torch.cuda.is_available() else False

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=WORKERS, pin_memory=pin)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=WORKERS, pin_memory=pin)
        
    # determine the device for training (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # instantiate the U-Net model and move it to the specified device
    model = UNet().to(device)
       
    # if FINETUNING:
    #     print("\nFinetuning...")
    #     model = torch.hub.load("mateuszbuda/brain-segmentation-pytorch", "unet", 
    #                            in_channels=3, out_channels=1, init_features=32, pretrained=True)
    #     model = model.to(device)

    # define the optimizer for training the model
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, 
                                 betas=(0.9, 0.999), weight_decay=L2_REG)
    
    # define learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-5, verbose=True)
    
    if RESUME_TRAIN:
        print("\nLoading model...")
        model, optimizer, start_epoch, scheduler = load_checkpoint(fpath="./checkpoints/model.pt",
                                                                   model=model, optimizer=optimizer)
    
    # define the loss function for training the model
    loss_fn = nn.BCEWithLogitsLoss() # loss_fn = dc_loss 

    # create an instance of the Solver class for training and validation
    solver = Solver(epochs=EPOCHS,
                    start_epoch=start_epoch,
                    writer=None,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=loss_fn,
                    patience=PATIENCE)

    # train the neural network
    solver.train_net()

    # # check the model ability
    # solver.check_results()

    # # test the neural network
    # solver.test_model()
