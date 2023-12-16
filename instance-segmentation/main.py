import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler

from solver import Solver
from dataset import collate_fn
from models import create_model
from dataset import PennFudanPedDataset


# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# parameters
K_FOLDS = 5
EPOCHS = 200
RESIZE = 224
PATIENCE = 5
BATCH_SIZE = 2
RESUME_TRAIN = False
DATA_PATH = Path("./data/PennFudanPed")
RUN_NAME = "test_1"


if __name__ == "__main__":    
    # get classes
    classes = ["__background__", "PASpersonWalking"]
    
    # to store fold results
    results = {}

    # create instances of the custom dataset for training and testing
    dataset = PennFudanPedDataset(dir_path=DATA_PATH, resize_h=RESIZE, resize_w=RESIZE, 
                                  classes=classes, data_aug=False)
    
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, collate_fn=collate_fn, num_workers=1)

    # pin memory to speed up the host (CPU) to device (GPU) transfer
    pin = True if torch.cuda.is_available() else False

    # determine the device for training (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: ", device, "\n")

    # define the K-fold Cross Validator
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    # start print
    print("--------------------------------")

    # k-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        # sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        # define data loaders for training and testing data in this fold
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=pin, num_workers=os.cpu_count(), 
                                  sampler=train_subsampler, collate_fn=collate_fn)
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=pin, num_workers=os.cpu_count(), 
                                 sampler=test_subsampler, collate_fn=collate_fn)
        
        # initialize the model and move to the computation device
        model = create_model(num_classes=len(classes))        
        if RESUME_TRAIN:
            model.load_state_dict(torch.load("./checkpoints/model-epoch=3-val_loss=0.1809.pth", 
                                             map_location=device))
        model = model.to(device)

        # get the model parameters
        params = [p for p in model.parameters() if p.requires_grad]

        # define the optimizer
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005) 
        # optimizer = torch.optim.Adam(params=params, lr=0.001, betas=(0.9, 0.999))

        # create an instance of the Solver class for training and validation        
        solver = Solver(epochs=EPOCHS, 
                        device=device, 
                        train_loader=train_loader, 
                        valid_loader=test_loader, 
                        model=model, 
                        optimizer=optimizer,
                        patience=PATIENCE)

        # train the neural network
        solver.train_net()

        # # evaluate the model on the test set
        # model_results = solver.eval_model()
        # model_results["test_name"] = RUN_NAME

        # # print accuracy
        # print(f"\nAccuracy for fold {fold}: {model_results['model_acc'] * 100:.3f}%")
        # print("\n--------------------------------")
        # results[fold] = model_results  # dictionary of (dictionary) results

        # clean up memory
        del model, optimizer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # with open(f"./statistics/{RUN_NAME}-kfold-model_results.json", "w") as outfile:
    #     json.dump(results, outfile)

    # # print fold results
    # print(f"K-FOLD CROSS VALIDATION RESULTS FOR {K_FOLDS} FOLDS")
    # print("--------------------------------")
    # sum_acc = 0.0
    # sum_loss = 0.0

    # for i, fold_result in results.items():
    #     acc = fold_result["model_acc"]
    #     loss = fold_result["model_loss"]
    #     print(f"Fold {i}: Accuracy = {acc * 100:.3f}%, Loss = {loss}")
    #     sum_acc += acc
    #     sum_loss += loss

    # avg_acc = sum_acc / len(results)
    # avg_loss = sum_loss / len(results)

    # print(f"\nAverage Accuracy: {avg_acc * 100:.3f}%")
    # print(f"Average Loss: {avg_loss:.3f}\n")
