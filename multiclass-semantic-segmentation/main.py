import os
import torch
import random
import numpy as np
from glob import glob
import torch.nn as nn
from torchsummary import summary
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from pytorch_model_summary import summary
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.train import TrainEpoch
from segmentation_models_pytorch.utils.train import ValidEpoch

from models import UNet
from solver import Solver
from models import CustomLoss
from models import AttentionUNet
from dataset import BucherDataset
from models import LightAttentionUNet
from utils import calculate_class_weights
from early_stopping import load_checkpoint
# from attUnetGithub import AttentionUNet


# os.chdir("./drive/MyDrive/Workspace/UNet")

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
# ACTIVATION = None
RESUME_TRAIN = True
ENCODER = "resnet18"
TEST_NAME = "test_1"
ENCODER_WEIGHTS = None # "imagenet"
IMGS_PTH = "./data/preprocessed-images/**/*.png"
WORKERS = os.cpu_count() if os.cpu_count() < 4 else 4
L2_REG = 0 # 0.004


# main script
if __name__ == "__main__":
    # shuffle and split dataset into training and testing sets
    all_images = shuffle([fn for fn in glob(pathname=IMGS_PTH) if "_mask" not in fn], random_state=SEED)

    X_test = [filename for filename in all_images if "test" in filename]
    X_train = [filename for filename in all_images if "train" in filename]
    X_validation = [filename for filename in all_images if "validation" in filename]
    # X_train, X_test = train_test_split(all_images, test_size=0.2, random_state=SEED)

    print(f"\nTrain-set n-samples: {len(X_train)}, Test-set n-samples: {len(X_test)}, Validation-set n-samples: {len(X_validation)}")    
   
    # create datasets and apply data augmentation
    train_dataset = BucherDataset(images=X_train, resize_h=RESIZE, resize_w=RESIZE, data_aug=True)
    valid_dataset = BucherDataset(images=X_validation, resize_h=RESIZE, resize_w=RESIZE, data_aug=False)
    test_dataset = BucherDataset(images=X_test, resize_h=RESIZE, resize_w=RESIZE, data_aug=False)
    
    # dataLoader settings for efficient data loading
    pin = True if torch.cuda.is_available() else False
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=WORKERS, pin_memory=pin)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, 
                              shuffle=False, num_workers=WORKERS, pin_memory=pin)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=WORKERS, pin_memory=pin)

    # determine the device for training (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # instantiate the U-Net model and move it to the specified device
    # model = AttentionUNet(in_channels=3, n_classes=N_CLASSES).to(device)
    model = UNet(in_channels=3, n_classes=N_CLASSES).to(device)
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
        model, optimizer, START_EPOCH = load_checkpoint(fpath="./checkpoints/model-epoch=14-val_loss=0.4660.pt", # model-epoch=29-val_loss=0.1494.pt
                                                        model=model, optimizer=optimizer) # , scheduler
    
    # # define the loss function for training the model
    # # https://discuss.pytorch.org/t/loss-function-for-multi-class-semantic-segmentation/117570
    # # w = calculate_class_weights(train_loader, num_classes=3) # 0.00042003 0.006354 0.99323
    # # w[2] = w[1] # same weight for foliage and waste
    # # # w = torch.tensor([0.00042003, 0.006354, 0.008354], dtype=torch.float32, device=device)
    # # loss_fn = nn.CrossEntropyLoss(weight=w.to(device))
    loss_fn = smp.losses.JaccardLoss(mode="multiclass", classes=[0, 1, 2], from_logits=True)
    loss_fn.__name__ = "Jaccard_loss"
    # w = torch.tensor([0.00042003, 0.006354, 0.007354], dtype=torch.float32, device=device)
    # loss_fn = CustomLoss(n_classes=3, device=device, ce_weights=w, avg=True, W_CE=1., W_IoU=1., W_Dice=1.)

    # create an instance of the Solver class for training and validation
    solver = Solver(test_name=TEST_NAME,
                    epochs=EPOCHS,
                    start_epoch=START_EPOCH,
                    writer=None,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=test_loader,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=loss_fn,
                    patience=PATIENCE)
    
    # get input shape
    inputs, _ = next(iter(train_loader))
    input_size = inputs.shape[1:]

    print("\nModel: ")
    # summary(model=model, input_size=input_size)
    summary(model, torch.zeros(inputs.shape).to(device), max_depth=5, print_summary=True)

    # train the neural network
    solver.train_net()

    # test the neural network
    test_losses = []
    solver.valid_net(epoch=None, data_loader=test_loader, 
                     valid_losses=test_losses, collect_stats=True, show_results=True)
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
