import os
import torch
import random
import numpy as np
from glob import glob
import torch.nn as nn
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.train import TrainEpoch
from segmentation_models_pytorch.utils.train import ValidEpoch

from models import UNet
from solver import Solver
from dataset import BucherDataset
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
LR = 1e-3
RESIZE = 32
EPOCHS = 200
PATIENCE = 10
N_CLASSES = 3
BATCH_SIZE = 4
START_EPOCH = 0
ACTIVATION = None
RESUME_TRAIN = True
ENCODER = "resnet18"
ENCODER_WEIGHTS = None # "imagenet"
IMGS_PTH = "./data/images/*.png"
WORKERS = os.cpu_count() if os.cpu_count() < 4 else 4
L2_REG = 0 # 0.004


def calculate_class_weights(dataloader: DataLoader, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights based on inverse class frequencies in a dataset.

    Parameters:
        - dataloader (DataLoader): The DataLoader containing the dataset.
        - num_classes (int): The number of classes in the dataset.

    Returns:
        - torch.Tensor: A tensor containing the calculated class weights.
    """
    # initialize a tensor to store the count of samples for each class
    class_counts = torch.zeros(num_classes)

    # calculate class frequencies
    for _, masks in dataloader:
        for class_idx in range(num_classes):
            class_counts[class_idx] += torch.sum(masks == class_idx).item()

    # calculate inverse class frequencies, avoiding division by zero
    inverse_class_frequencies = torch.where(class_counts > 0, 1 / class_counts, 0)

    # normalize weights
    weights = inverse_class_frequencies / inverse_class_frequencies.sum()

    return weights

# main script
if __name__ == "__main__":
    # shuffle and split dataset into training and testing sets
    all_images = shuffle([fn for fn in glob(pathname=IMGS_PTH) if "_mask" not in fn], random_state=SEED)
    
    X_train, X_test = train_test_split(all_images, test_size=0.2, random_state=SEED)
    
    # create datasets and apply data augmentation
    train_dataset = BucherDataset(images=X_train, resize_h=RESIZE, resize_w=RESIZE, data_aug=True)
    test_dataset = BucherDataset(images=X_test, resize_h=RESIZE, resize_w=RESIZE, data_aug=False)
    
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
    model = UNet(n_classes=N_CLASSES).to(device)
    # model = smp.UnetPlusPlus(encoder_name=ENCODER, 
    #                          encoder_weights=ENCODER_WEIGHTS, 
    #                          in_channels=3, classes=3, activation=ACTIVATION)
       
    # define the optimizer for training the model
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, 
                                 betas=(0.9, 0.999), weight_decay=L2_REG)
    
    # define learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=1e-7, verbose=True)
    
    if RESUME_TRAIN:
        print("\nLoading model...")
        model, optimizer, START_EPOCH = load_checkpoint(fpath="./checkpoints/model-epoch=29-val_loss=0.1494.pt",
                                                        model=model, optimizer=optimizer) # , scheduler
    
    # define the loss function for training the model
    # https://discuss.pytorch.org/t/loss-function-for-multi-class-semantic-segmentation/117570
    w = calculate_class_weights(train_loader, num_classes=3) # 0.00042003 0.006354 0.99323
    w[2] = w[1] # same weight for foliage and waste
    # w = torch.tensor([0.00042003, 0.006354, 0.008354], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device))
    # loss_fn = smp.losses.DiceLoss(mode="multiclass")
    # loss_fn.__name__ = "Dice_loss"

    # create an instance of the Solver class for training and validation
    solver = Solver(epochs=EPOCHS,
                    start_epoch=START_EPOCH,
                    writer=None,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=loss_fn,
                    patience=PATIENCE)

    # # train the neural network
    # solver.train_net()

    # # check the model ability
    # solver.check_results()

    # test the neural network
    test_losses = []
    solver.valid_net(epoch=None, valid_losses=test_losses, show_results=True)
    test_loss = np.average(test_losses)
    print(f"Mean test loss: {np.mean(test_losses):.4f}\n")


    # SMP - Package
    ###############################################################################################
    # metrics = [IoU(threshold=0.5)]
    # # create epoch runners 
    # # it is a simple loop of iterating over dataloader`s samples
    # train_epoch = TrainEpoch(
    #     model, 
    #     loss=loss_fn, 
    #     metrics=metrics, 
    #     optimizer=optimizer,
    #     device=device,
    #     verbose=True,
    # )
    # valid_epoch = ValidEpoch(
    #     model, 
    #     loss=loss_fn, 
    #     metrics=metrics, 
    #     device=device,
    #     verbose=True,
    # )    
    # max_score = 0
    # for i in range(START_EPOCH, EPOCHS):        
    #     print('\nEpoch: {}'.format(i))
    #     train_logs = train_epoch.run(train_loader)
    #     valid_logs = valid_epoch.run(test_loader)        
    #     # do something (save model, change lr, etc.)
    #     if max_score < valid_logs['iou_score']:
    #         max_score = valid_logs['iou_score']
    #         torch.save(model, './best_model.pth')
    #         print('Model saved!')            
    #     if i == 25:
    #         optimizer.param_groups[0]['lr'] = 1e-5
    #         print('Decrease decoder learning rate to 1e-5!')
    ###############################################################################################
