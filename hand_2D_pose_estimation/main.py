import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import UNet
from solver import Solver
from models import IoULoss
from utils import plot_image
from dataset import FreiHandDataset
from dataset import create_dataframe
from utils import get_keypoint_location
from early_stopping import load_checkpoint
from utils import draw_keypoints_connection

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# hyperparameters
LR = 0.1
RESIZE = 128
EPOCHS = 200
PATIENCE = 10
BATCH_SIZE = 4
start_epoch = 0
N_KEYPOINTS = 21
L2_REG = 0 # 0.004
FINETUNING = False
RESUME_TRAIN = False
ROOT = "./data/FreiHAND_pub_v2"
WORKERS = os.cpu_count() if os.cpu_count() < 4 else 4

# main script
if __name__ == "__main__":
    # create a dataframe
    df = create_dataframe(root=ROOT)

    # split the data into training, validation, and test sets
    X_train, X_val_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=SEED)
    X_valid, X_test = train_test_split(X_val_test, test_size=0.05, shuffle=True, random_state=SEED)

    # create datasets
    train_dataset = FreiHandDataset(df=X_train, resize=RESIZE, n_keypoints=N_KEYPOINTS)
    valid_dataset = FreiHandDataset(df=X_valid, resize=RESIZE, n_keypoints=N_KEYPOINTS)
    test_dataset = FreiHandDataset(df=X_test, resize=RESIZE, n_keypoints=N_KEYPOINTS)

    # dataLoader settings for efficient data loading
    pin = True if torch.cuda.is_available() else False

    # create dataloaders for batch training
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=WORKERS, pin_memory=pin)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=WORKERS, pin_memory=pin)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, 
                             shuffle=False, num_workers=WORKERS, pin_memory=pin)
    
    # DEBUG
    for i, data in enumerate(train_loader):
        for image, heatmaps, keypoints in zip(data["image"], data["heatmaps"], data["keypoints"]):
            # get (x, y) keypoints coordinates
            x_locations, y_locations = get_keypoint_location(heatmaps=heatmaps)
            # for x, y in zip(x_locations, y_locations):
            #     print(f"Estimated (x, y) location: ({int(x.item())}, {int(y.item())})")
            # # visualize the heatmaps and keypoints
            # for heatmap, x, y in zip(heatmaps, x_locations, y_locations):
            #     plt.imshow(heatmap.squeeze(), cmap="gray")
            #     plt.scatter(int(x.item()), int(y.item()), c="red", marker="x")
            #     plt.show()
            # # visualize image, heatmaps and keypoints
            # plot_image(image=image, heatmaps=heatmaps,
            #            keypoints=keypoints, resize=RESIZE)
            # viosualize image and keypoint connections
            draw_keypoints_connection(image=image, uv_coords=(x_locations, y_locations))
        
    # # determine the device for training (use GPU if available, otherwise use CPU)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"\nDevice: {device}")

    # # instantiate the U-Net model and move it to the specified device
    # model = UNet(in_channels=3, out_channels=N_KEYPOINTS).to(device)

    # # define the optimizer for training the model
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, 
    #                              betas=(0.9, 0.999), weight_decay=L2_REG)
    
    # # define learning rate scheduler
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-7, verbose=True)
    
    # # resume training from a checkpoint if specified
    # if RESUME_TRAIN:
    #     print("\nLoading model...")
    #     model, optimizer, start_epoch, scheduler = load_checkpoint(fpath="./checkpoints/model.pt",
    #                                                                model=model, optimizer=optimizer)
    
    # # define the loss function for training the model
    # loss_fn = IoULoss()

    # # create an instance of the Solver class for training and validation
    # solver = Solver(epochs=EPOCHS,
    #                 start_epoch=start_epoch,
    #                 writer=None,
    #                 train_loader=train_loader,
    #                 test_loader=test_loader,
    #                 device=device,
    #                 model=model,
    #                 optimizer=optimizer,
    #                 scheduler=scheduler,
    #                 criterion=loss_fn,
    #                 patience=PATIENCE)

    # # train the neural network
    # solver.train_net()
