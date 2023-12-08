########################################################################################################################################
# source: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md #
#######################################################################################################################################

import os
import json
import torch
import random
# import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torchvision import  models
# from torchvision import transforms
# import pytorch_model_summary as pms
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchvision.models import ResNet18_Weights
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from solver import Solver
from custom_dataset import CleanDirtyRoadDataset


# reproducibility
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


def kfold_main(args): 
    base_img_pth = Path("./data/Images")
    metadata_path = Path("./data/metadata.csv")

    # configuration options
    k_folds = 5

    # For fold results
    results = {}

    # Prepare MNIST dataset by concatenating Train/Test part; we split later.
    df = pd.read_csv(filepath_or_buffer=metadata_path)
    
    resize = 224

    # # define data transformations for training and testing
    # if args.apply_transformations:
    #     train_transform = transforms.Compose([transforms.Resize(size=(resize, resize)),
    #                                           transforms.RandomHorizontalFlip(p=0.5),
    #                                           transforms.RandomVerticalFlip(p=0.5),
    #                                           transforms.RandomRotation(degrees=15),
    #                                           transforms.ToTensor(),
    #                                           transforms.Normalize([0.485, 0.456, 0.406], 
    #                                                                [0.229, 0.224, 0.225])])
    #     test_transform = transforms.Compose([transforms.Resize(size=(resize, resize)),
    #                                          transforms.ToTensor(),
    #                                          transforms.Normalize([0.485, 0.456, 0.406],
    #                                                               [0.229, 0.224, 0.225])])
    # else:
    #     train_transform = None
    #     test_transform = None

    # create instances of the custom dataset for training and testing
    dataset = CleanDirtyRoadDataset(X=df["filename"], y=df["label"], 
                                    resize=resize, data_root=base_img_pth, transform=None)
    
    # pin_memory: speed up the host (cpu) to device (gpu) transfer
    pin = True if torch.cuda.is_available() else False   

    # determine the device for training (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: ", device, "\n")

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # Start print
    print("--------------------------------")

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # tensorboard specifications
        date =  "_" + datetime.now().strftime("%d%m%Y-%H%M%S")
        writer = SummaryWriter("./runs/" + args.run_name + f"-fold-{fold}" + date)

        # Print
        print(f"FOLD {fold}")
        print("--------------------------------")

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=pin,
                                  num_workers=os.cpu_count(), sampler=train_subsampler)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=pin, 
                                 num_workers=os.cpu_count(), sampler=test_subsampler)

        # Init the neural network
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # freeze all params (fine-tuning: feature learning)
        for param in model.parameters():
            param.requires_grad = False
        
        # add a new final layer
        # number of input features of last layer
        nr_filters = model.fc.in_features
        model.fc = nn.Linear(nr_filters, args.num_classes)
        # # to avoid weight leaks
        # ###########################
        # model.fc.reset_parameters()
        # ###########################
        
        model = model.to(device)

        # # to avoid weight leaks
        # ###########################
        # model.apply(reset_weights)
        # ###########################

        # images, _ = next(iter(train_loader))
        # images = images.to(device)
        # img_grid = torchvision.utils.make_grid(images)
        # writer.add_image("sample-images", img_grid, global_step=0)

        # print("\nModel: ")
        # pms.summary(model, torch.zeros(images.shape, device=device), max_depth=5, print_summary=True)

        # define the optimizer and loss function for training the model
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=args.lr, betas=(0.9, 0.999))
        loss_fn = nn.BCEWithLogitsLoss()

        # create an instance of the Solver class for training and validation
        solver = Solver(epochs=args.num_epochs, 
                        writer=writer,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=device,
                        model=model,
                        optimizer=optimizer,
                        criterion=loss_fn,
                        patience=args.patience)
        
        # train the neural network
        solver.train_net()

        model_results = solver.eval_model()
        model_results["test_name"] = args.run_name

        # Print accuracy
        print(f"\nAccuracy for fold {fold}: {model_results['model_acc'] * 100:.3f}%")
        print("\n--------------------------------")
        results[fold] = model_results # dict of dicts

    with open(f"./statistics/{args.run_name}-kfold-model_results.json", "w") as outfile:
        json.dump(results, outfile)

    # Print fold results
    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS")
    print("--------------------------------")
    sum_acc = 0.0
    sum_loss = 0.0

    for i, fold_result in results.items():
        acc = fold_result["model_acc"]
        loss = fold_result["model_loss"]
        print(f"Fold {i}: Accuracy = {acc * 100:.3f}%, Loss = {loss}")
        sum_acc += acc
        sum_loss += loss

    avg_acc = sum_acc / len(results)
    avg_loss = sum_loss / len(results)

    print(f"\nAverage Accuracy: {avg_acc * 100:.3f}%")
    print(f"Average Loss: {avg_loss:.3f}\n")
