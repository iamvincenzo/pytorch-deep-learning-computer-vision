import torch
import numpy as np
from tqdm import tqdm

from early_stopping import EarlyStopping


class Solver(object):
    def __init__(self, epochs, device, train_loader, valid_loader, model, optimizer, patience):
        """
        Initialize the Solver object.

        Parameters:
            - epochs (int): Number of training epochs.
            - device (torch.device): The computation device (GPU or CPU).
            - train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            - valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            - model (torch.nn.Module): The neural network model.
            - optimizer (torch.optim.Optimizer): The optimizer for model training.
        """
        super(Solver, self).__init__()
        self.model = model
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.patience = patience

    def train_net(self):
        """
        Train the neural network using the provided data loaders.
        """
        print(f"\nStarting training...")
        
        # lists to track training and validation losses
        train_losses = []
        valid_losses = []
        # lists to track average losses per epoch
        avg_train_losses = []
        avg_valid_losses = []

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        
        # see self.valid_net() to understand why .train() mode is disabled
        # self.model.train()

        for epoch in range(self.epochs): 
            print(f"\nTraining iteration | Epoch[{epoch + 1}/{self.epochs}]")

            # use tqdm for a progress bar during training
            loop = tqdm(iterable=enumerate(self.train_loader), 
                        total=len(self.train_loader), 
                        leave=True)
            
            for batch_idx, (x_train, y_train) in loop:
                # move input data and target labels to the specified device (GPU or CPU)
                x_train = list(image.to(self.device) for image in x_train)
                y_train = [{k: v.to(self.device) for k, v in t.items()} for t in y_train]

                # forward pass and compute loss
                loss_dict = self.model(x_train, y_train)
                losses = sum(loss for loss in loss_dict.values())
                
                # backward pass and optimization
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                train_losses.append(losses.item())
                
                # update the loss value beside the progress bar for each iteration
                loop.set_description(desc=f"Batch {batch_idx}, Loss: {losses.item():.3f}")

            loop.close()

            # validate the model after each epoch
            self.valid_net(valid_losses=valid_losses)

            train_loss = np.mean(train_losses) 
            valid_loss = np.mean(valid_losses)  
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.epochs}] | train-loss: {train_loss:.4f} | "
                  f"validation-loss: {valid_loss:.4f}")

            train_losses = []
            valid_losses = []   

            # early stopping checks for improvement in validation loss
            early_stopping(epoch, valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping...")
                break

        print("\nTraining model Done...")

    def valid_net(self, valid_losses):
        """
        Validate the neural network using the provided validation data loader.
        """
        print(f"\nStarting validation...\n")
        
        # self.model.eval()
        '''
            It is not necessary to put the model in .eval()/.train() mode because otherwise Faster R-CNN
            in .eval() mode is not able to calculate the loss during validation. What could be done
            is to put only BatchNorm1d/2d/3d and Dropout layers in .eval() mode since they behave differently 
            in eval mode, while leaving the entire model in training mode for the calculation of the loss 
            during validation. However, it is not necessary as the PyTorch implementations of Faster R-CNN do 
            not use Dropout, and FrozenBatchNorm2d is used, which has frozen statistics and behaves the 
            same way in both training and validation.            
            Credits:
                - https://discuss.pytorch.org/t/how-to-calculate-validation-loss-for-faster-rcnn/96307/11
                - https://discuss.pytorch.org/t/how-to-calculate-validation-loss-for-faster-rcnn/96307/12
                - https://discuss.pytorch.org/t/how-to-calculate-validation-loss-for-faster-rcnn/96307/16

            with torch.no_grad():
                Disabling gradient calculation when you are sure that you will not call Tensor.backward().
                In this mode, the result of every computation will have requires_grad=False, even when the
                inputs have requires_grad=True.
            model.eval():
                will notify all your layers that you are in eval mode, that way, batchnorm or dropout
                layers will work in eval mode instead of training mode.            
            Credits:
                - https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
        '''

        with torch.no_grad():
            # use tqdm for a progress bar during validation
            loop = tqdm(iterable=enumerate(self.valid_loader), 
                        total=len(self.valid_loader), 
                        leave=True)
            
            for batch_idx, (x_valid, y_valid) in loop:
                # move input data and target labels to the specified device (GPU or CPU)
                x_valid = list(image.to(self.device) for image in x_valid)
                y_valid = [{k: v.to(self.device) for k, v in t.items()} for t in y_valid]

                # forward pass and compute loss
                loss_dict = self.model(x_valid, y_valid)

                losses = sum(loss for loss in loss_dict.values())
                
                valid_losses.append(losses.item())
                
                # update the loss value beside the progress bar for each iteration
                loop.set_description(desc=f"Validation Loss: {losses.item():.3f}")

            loop.close()

        # self.model.train()
