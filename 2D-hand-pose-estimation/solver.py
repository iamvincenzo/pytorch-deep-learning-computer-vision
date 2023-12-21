import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from early_stopping import EarlyStopping


class Solver(object):
    def __init__(self, epochs, start_epoch, writer, train_loader, test_loader, device, model, optimizer, scheduler, criterion, patience):
        """
        A class to handle training and validation of a PyTorch neural network.

        Args:
            - epochs (int): Number of training epochs.
            - writer (object): Object for writing logs (e.g., TensorBoard writer).
            - train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            - test_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            - device (torch.device): Device on which to perform training and validation.
            - model (torch.nn.Module): Neural network model.
            - optimizer (torch.optim.Optimizer): Optimization algorithm.
            - criterion (torch.nn.Module): Loss function.
            - patience (int): Number of epochs to wait for improvement before early stopping.
        
        Returns:
            - None
        """
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.writer = writer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.patience = patience

        self.model_reslts = {
            "test_name": "",
            "model_name": self.model.__class__.__name__,
            "train_time": 0.0,
            "tot_epochs": 0,
            "model_loss": 0.0,
            "model_acc": 0.0
        }

    def train_net(self):
        """
        Trains the neural network using the specified DataLoader for training data.

        This method also performs early stopping based on the validation loss.

        Prints training and validation statistics for each epoch.
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

        self.model.train()

        # loop over the dataset multiple times
        for epoch in range(self.start_epoch, self.epochs):
            print(f"\nTraining iteration | Epoch[{epoch + 1}/{self.epochs}]")

            # use tqdm for a progress bar during training
            loop = tqdm(iterable=enumerate(self.train_loader),
                        total=len(self.train_loader),
                        leave=True)
            
            all_masks = torch.tensor([], device=self.device)
            all_preds = torch.tensor([], device=self.device)

            # loop over training data
            for batch_idx, data in loop:
                # # used to check model improvements during the training
                # if batch_idx == 0:
                #     self.check_results()

                # move data and labels to the specified device
                x_train = data["image"].to(self.device)
                y_train = data["heatmaps"].to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                # logits --> probabilities --> labels
                logits = self.model(x_train)
                probs = torch.sigmoid(logits)
                y_pred = torch.round(probs)

                # calculate the loss
                loss = self.criterion(y_pred, y_train)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                self.optimizer.step()

                # record training loss
                train_losses.append(loss.item())

                all_preds = torch.cat([all_preds, y_pred], dim=0)
                all_masks = torch.cat([all_masks, y_train], dim=0)

                # update the loss value beside the progress bar for each iteration
                loop.set_description(desc=f"Batch {batch_idx}, Loss: {loss.item():.3f}")

            loop.close()

            print(self.compute_metrics(preds=all_preds, masks=all_masks))

            # validate the model on the validation set
            self.valid_net(epoch=epoch, valid_losses=valid_losses)

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # step should be called after validate
            self.scheduler.step(valid_loss)

            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.epochs}] | train-loss: {train_loss:.4f} | "
                  f"validation-loss: {valid_loss:.4f}")

            if self.writer is not None:
                self.writer.add_scalar("training-loss", train_loss,
                                       epoch * len(self.train_loader) + batch_idx)
                self.writer.add_scalar("validation-loss", valid_loss,
                                       epoch * len(self.test_loader) + batch_idx)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []    

            checkpoint = {
                "epoch": epoch, # + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            }       

            # early stopping checks for improvement in validation loss
            early_stopping(checkpoint=checkpoint, val_loss=valid_loss)

            if early_stopping.early_stop:
                print("Early stopping...")
                break

        print("\nTraining model Done...")

        self.model_reslts["tot_epochs"] = epoch + 1

        if self.writer is not None:
            # write all remaining data in the buffer
            self.writer.flush()
            # free up system resources used by the writer
            self.writer.close()

    def valid_net(self, epoch, valid_losses):
        """
        Validates the neural network on the specified DataLoader for validation data.

        Records validation losses in the provided list.

        Args:
            epoch (int): The current epoch during validation.
            valid_losses (list): List to record validation losses.
        """
        print(f"\nStarting validation...\n")

        all_masks = torch.tensor([], device=self.device)
        all_preds = torch.tensor([], device=self.device)

        self.model.eval()

        # no need to calculate the gradients for outputs
        with torch.no_grad():
            loop = tqdm(iterable=enumerate(self.test_loader),
                        total=len(self.test_loader),
                        leave=True)

            for batch_idx, data in loop:
                # move data and labels to the specified device
                x_valid = data["image"].to(self.device)
                y_valid = data["heatmaps"].to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                # logits --> probabilities --> labels
                logits = self.model(x_valid)
                probs = torch.sigmoid(logits)
                y_pred = torch.round(probs)

                # calculate the loss
                loss = self.criterion(probs, y_valid)

                # record validation loss
                valid_losses.append(loss.item())                

                all_preds = torch.cat([all_preds, y_pred], dim=0)
                all_masks = torch.cat([all_masks, y_valid], dim=0)

                # update the loss value beside the progress bar for each iteration
                loop.set_description(desc=f"Batch {batch_idx}, Loss: {loss.item():.3f}")

            loop.close()

            print(self.compute_metrics(preds=all_preds, masks=all_masks))

        # set the model back to training mode
        self.model.train()

    def compute_metrics(self, preds, masks):
        """
        Compute evaluation metrics including accuracy, precision, recall, and F1 score.

        Args:
            - mask (torch.Tensor): Ground truth mask.
            - pred (torch.Tensor): Predicted mask.

        Returns:
            - dict: Dictionary containing computed metrics.
        """
        tp = torch.sum((preds == 1) & (masks == 1)).item()
        fp = torch.sum((preds == 1) & (masks == 0)).item()
        fn = torch.sum((preds == 0) & (masks == 1)).item()
        tn = torch.sum((preds == 0) & (masks == 0)).item()

        n = 1e-20

        accuracy = (tp + tn) / (tp + fp + fn + tn + n)
        precision = tp / (tp + fp + n)
        recall = tp / (tp + fn + n)
        f1_score = 2 * (precision * recall) / (precision + recall + n)

        return {"accuracy": accuracy, "precision": precision,
                "recall": recall, "f1_score": f1_score}
