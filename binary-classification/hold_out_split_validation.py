import os
import json
import torch
import random
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torchvision import models
from torchvision import transforms
import pytorch_model_summary as pms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from solver import Solver
from custom_dataset import CleanDirtyRoadDataset

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# parameters
RESIZE = 224

def holdout_main(args):
    """
    Main function for the holdout validation approach.

    Args:
        args (Namespace): Command-line arguments parsed by argparse.

    Returns:
        None
    """
    # tensorboard specifications
    date = "_" + datetime.now().strftime("%d%m%Y-%H%M%S")
    writer = SummaryWriter("./runs/" + args.run_name + date)

    base_img_pth = Path("./data/Images")
    metadata_path = Path("./data/metadata.csv")

    df = pd.read_csv(filepath_or_buffer=metadata_path)

    X_train, X_test, y_train, y_test = train_test_split(df["filename"], df["label"],
                                                        test_size=0.2, random_state=SEED, shuffle=True)

    # define data transformations for training and testing
    if args.apply_transformations:
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
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=os.cpu_count(), pin_memory=pin)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=os.cpu_count(), pin_memory=pin)

    # determine the device for training (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: ", device)

    finetuning = True

    if finetuning:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # freeze all params
        for param in model.parameters():
            param.requires_grad = False

        # add a new final layer
        nr_filters = model.fc.in_features
        model.fc = nn.Linear(nr_filters, args.num_classes)
        model = model.to(device)
    # else:
    #     # create an instance of the model and move it to the specified device
    #     model = BinaryImageClassifier(output_size=args.num_classes).to(device)

    images, _ = next(iter(train_loader))
    images = images.to(device)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image("sample-images", img_grid, global_step=0)

    print("\nModel: ")
    pms.summary(model, torch.zeros(images.shape, device=device), max_depth=5, print_summary=True)

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

    # collect statistics
    model_results = solver.eval_model()
    model_results["test_name"] = args.run_name
    with open(f"./statistics/{args.run_name}-holdout-model_results.json", "w") as outfile:
        json.dump(model_results, outfile)
