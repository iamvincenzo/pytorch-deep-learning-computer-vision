import os
import torch

class EarlyStopping(object):
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    
    Credits:
        Copyright (c) 2018 Bjarte Mehus Sunde
    
    Args:
        - patience (int): How long to wait after the last time validation loss improved.
                        Default: 5
        - delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
        - path (str): Path for the checkpoint to be saved to.
                        Default: './checkpoints/'
        - verbose (bool): If True, prints a message for each validation loss improvement.
                        Default: False
    """
    def __init__(self, patience=5, delta=0, path="./checkpoints/", verbose=False):
        # initialize the parameters
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.best_model_path = None

        # counter to track the number of epochs without improvement
        self.counter = 0

        # best validation score initialized to None
        self.best_score = None

        # flag to indicate if early stopping criteria are met
        self.early_stop = False

        # minimum validation loss initialized to positive infinity
        self.val_loss_min = float('inf')

        # create directory if not exists
        os.makedirs(self.path, exist_ok=True)

    def __call__(self, epoch, val_loss, model):
        """
        Call method to evaluate the validation loss and perform early stopping.
        
        Args:
            - epoch (int): Current epoch number.
            - val_loss (float): Validation loss value.
            - model (torch.nn.Module): PyTorch model to be saved.

        Returns:
            - None.
        """
        # check if it's the first validation, or there is an improvement, or no improvement
        if self.best_score is None:
            self._handle_first_validation(epoch, val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self._handle_no_improvement()
        else:
            self._handle_improvement(epoch, val_loss, model)

    def _handle_first_validation(self, epoch, val_loss, model):
        """
        Handle the case of the first validation.

        Args:
            - epoch (int): Current epoch number.
            - val_loss (float): Validation loss value.
            - model (torch.nn.Module): PyTorch model to be saved.
        """
        self.best_score = val_loss
        self.save_checkpoint(epoch, val_loss, model)

    def _handle_no_improvement(self):
        """
        Handle the case of no improvement in validation loss.
        """
        self.counter += 1
        print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            # if the counter exceeds patience, set the early_stop flag
            self.early_stop = True

    def _handle_improvement(self, epoch, val_loss, model):
        """
        Handle the case of an improvement in validation loss.

        Args:
            - epoch (int): Current epoch number.
            - val_loss (float): Validation loss value.
            - model (torch.nn.Module): PyTorch model to be saved.
        """
        self.counter = 0
        self.best_score = val_loss
        self.save_checkpoint(epoch, val_loss, model)

    def save_checkpoint(self, epoch, val_loss, model):
        """
        Saves model when validation loss decreases.

        Args:
            - epoch (int): Current epoch number.
            - val_loss (float): Validation loss value.
            - model (torch.nn.Module): PyTorch model to be saved.
        """
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        
        # remove the previous best model checkpoint if it exists
        if self.best_model_path is not None and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)

        # save the current best model's state_dict
        self.best_model_path = os.path.join(self.path, f"model-epoch={epoch + 1}-val_loss={val_loss:.4f}.pt")
        torch.save(obj=model.state_dict(), f=self.best_model_path)

        self.val_loss_min = val_loss
