import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
# from torchmetrics.classification import MulticlassJaccardIndex

from plotting_utils import show_preds
from early_stopping import EarlyStopping
# from plotting_utils import show_batch_preds
# from plotting_utils import show_batch_preds_with_transparency


class Solver(object):
    def __init__(self, epochs: int, start_epoch: int, writer: Any, train_loader: DataLoader, test_loader: DataLoader, 
                 device: torch.device, model: nn.Module, optimizer: optim.Optimizer, scheduler: LRScheduler, criterion: nn.Module, patience: int) -> None:
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
            - None.
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
        self.criterion = criterion.to(self.device)
        self.patience = patience

        self.model_reslts = {
            "test_name": "",
            "model_name": self.model.__class__.__name__,
            "train_time": 0.0,
            "tot_epochs": 0,
            "model_loss": 0.0,
            "model_acc": 0.0
        }

    def get_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int, smooth: Optional[float] = 2.22e-16):
        """
        Calculate IoU, dice coefficient, recall, for single class and the mean of the classes.

        Args:
            - y_true: target masks (n_samples, h, w), where each pixel is represented with the class id.
            - y_pred: predicted masks (n_samples, h, w).
            - num_classes: number of classes.

        Returns:
            - mIoU: mean IoU.
            - IoU_classes: list of IoU on classes.
            - mean_dice_coeff: mean dice coefficient.
            - dice_classes: list of dice coefficient on classes.
            - macro_recall: mean recall.
            - recall_classes: list of recall on classes.
        """
        IoU_classes = []
        dice_classes = []
        recall_classes = []

        for class_id in range(num_classes):
            TP = torch.sum((y_true == class_id) & (y_pred == class_id))
            true_labels = torch.sum(y_true == class_id)
            true_preds = torch.sum(y_pred == class_id)

            union = true_labels + true_preds - TP
            IoU_classes.append((TP / (union + smooth)).item())
            dice_classes.append((2 * TP / (true_labels + true_preds + smooth)).item())
            recall_classes.append((TP / (true_labels + smooth)).item())

        mIoU = round(sum(IoU_classes) / len(IoU_classes), 3)                        
        # metric = MulticlassJaccardIndex(num_classes=3, average="macro")
        # mIoU = metric(y_pred, y_true)
        
        IoU_classes = np.round(IoU_classes, 3)
        # metric = MulticlassJaccardIndex(num_classes=3, average="none")
        # IoU_classes = metric(y_pred, y_true)

        mean_dice_coeff = round(sum(dice_classes)/len(dice_classes), 3)
        dice_classes = np.round(dice_classes, 3)

        macro_recall = round(sum(recall_classes)/len(recall_classes), 3)
        recall_classes = np.round(recall_classes, 3)

        return mIoU, IoU_classes, mean_dice_coeff, dice_classes, macro_recall, recall_classes

    def check_results(self) -> None:
        """
        Method used to visualize show some samples at the first batch
        of each epoch to check model improvements.

        Args:
            - None.

        Returns:
            - None. 
        """
        self.model.eval()

        with torch.no_grad():
            # choose a random batch from the validation set
            images, masks = random.choice(list(self.test_loader))

            # move data and labels to the specified device
            images = images.to(self.device)
            masks = masks.to(self.device)

            # forward pass: compute predicted outputs by passing inputs to the model
            logits = self.model(images)
            
            # since we are using CrossEntropyLoss
            # logits --> probabilities --> labels
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            show_preds(images=images, masks=masks, preds=preds, alpha=None)
            # show_batch_preds(images=images, masks=masks, preds=y_pred)
            # show_batch_preds_with_transparency(images=images, masks=masks, preds=preds, alpha=None)

        self.model.train()

    def train_net(self) -> None:
        """
        Trains the neural network using the specified DataLoader for training data.

        This method also performs early stopping based on the validation loss.

        Prints training and validation statistics for each epoch.

        Args:
            - None.

        Returns:
            - None.
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
            for batch_idx, (x_train, y_train) in loop:
                # used to check model improvements during the training
                if batch_idx == 0:
                    self.check_results()

                # move data and labels to the specified device
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                logits = self.model(x_train)

                # since we are using CrossEntropyLoss
                # logits --> probabilities --> labels
                probs = torch.softmax(logits, dim=1)
                y_pred = torch.argmax(probs, dim=1)

                # calculate the loss
                loss = self.criterion(logits, y_train.squeeze(1))

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                self.optimizer.step()

                # record training loss
                train_losses.append(loss.item())

                all_preds = torch.cat([all_preds, y_pred], dim=0)
                all_masks = torch.cat([all_masks, y_train.squeeze()], dim=0)

                # update the loss value beside the progress bar for each iteration
                loop.set_description(desc=f"Batch {batch_idx}, Loss: {loss.item():.3f}")

            loop.close()

            mIoU, IoU_classes, mean_dice_coeff, dice_classes, macro_recall, recall_classes = self.get_metrics(y_true=all_masks,
                                                                                                              y_pred=all_preds,
                                                                                                              num_classes=3)
            print(f"\nmIoU: {mIoU}, mean_dice_coeff: {mean_dice_coeff}, macro_recall: {macro_recall}")            
            print(f"IoU_classess: {IoU_classes}")
            print(f"dice_classes: {dice_classes}")
            print(f"recall_classes: {recall_classes}\n")

            # validate the model on the validation set
            self.valid_net(epoch=epoch, valid_losses=valid_losses, show_results=False)

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            if self.scheduler is not None:
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
                # "scheduler_state_dict": self.scheduler.state_dict(),
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

    def valid_net(self, epoch: int, valid_losses: list, show_results: Optional[bool] = False) -> None:
        """
        Validates the neural network on the specified DataLoader for validation data.

        Records validation losses in the provided list.

        Args:
            - epoch (int): The current epoch during validation.
            - valid_losses (list): List to record validation losses.

        Returns:
            - None.
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

            for batch_idx, (x_valid, y_valid) in loop:
                # move data and labels to the specified device
                x_valid = x_valid.to(self.device)
                y_valid = y_valid.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                logits = self.model(x_valid)

                # since we are using CrossEntropyLoss
                # logits --> probabilities --> labels
                probs = torch.softmax(logits, dim=1)
                y_pred = torch.argmax(probs, dim=1)

                # calculate the loss
                loss = self.criterion(logits, y_valid.squeeze(1))

                # record validation loss
                valid_losses.append(loss.item())
    
                all_preds = torch.cat([all_preds, y_pred], dim=0)
                all_masks = torch.cat([all_masks, y_valid.squeeze()], dim=0)

                # update the loss value beside the progress bar for each iteration
                loop.set_description(desc=f"Batch {batch_idx}, Loss: {loss.item():.3f}")

                if show_results:
                    show_preds(images=x_valid, masks=y_valid, preds=y_pred, alpha=None)
                    # show_batch_preds(images=x_valid, masks=y_valid, preds=y_pred)
                    # show_batch_preds_with_transparency(images=x_valid, masks=y_valid, preds=y_pred, alpha=None)

            loop.close()

            mIoU, IoU_classes, mean_dice_coeff, dice_classes, macro_recall, recall_classes = self.get_metrics(y_true=all_masks,
                                                                                                              y_pred=all_preds,
                                                                                                              num_classes=3)
            print(f"\nmIoU: {mIoU}, mean_dice_coeff: {mean_dice_coeff}, macro_recall: {macro_recall}")            
            print(f"IoU_classess: {IoU_classes}")
            print(f"dice_classes: {dice_classes}")
            print(f"recall_classes: {recall_classes}\n")

        # set the model back to training mode
        self.model.train()
