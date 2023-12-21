import os
import torch


def load_checkpoint(fpath, model, optimizer, scheduler=None):
    """
    Load the model and optimizer state from a checkpoint file.

    Parameters:
        - fpath (str): File path to the checkpoint containing model and optimizer state.
        - model (torch.nn.Module): The PyTorch model to which the saved state will be loaded.
        - optimizer (torch.optim.Optimizer): The PyTorch optimizer to which the saved state will be loaded.
        - scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The PyTorch scheduler to which the saved state will be loaded.

    Returns:
        - model (torch.nn.Module): The model with loaded state.
        - optimizer (torch.optim.Optimizer): The optimizer with loaded state.
        - start_epoch (int): The starting epoch from the checkpoint.
        - scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler with loaded state if provided.
    """
    checkpoint = torch.load(fpath)

    start_epoch = checkpoint["epoch"]

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return model, optimizer, start_epoch, scheduler


class EarlyStopping(object):
    def __init__(self, patience=5, delta=0, path="./checkpoints", verbose=False):
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
                              Default: './checkpoints'
            - verbose (bool): If True, prints a message for each validation loss improvement.
                              Default: False
        
        Returns:
            - None.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.best_model_path = None

        # Counter to track the number of epochs without improvement
        self.counter = 0

        # Best validation score initialized to None
        self.best_score = None

        # Flag to indicate if early stopping criteria are met
        self.early_stop = False

        # Minimum validation loss initialized to positive infinity
        self.val_loss_min = float("inf")

        # Create directory if not exists
        os.makedirs(self.path, exist_ok=True)

    def __call__(self, checkpoint, val_loss):
        """
        Call method to evaluate the validation loss and perform early stopping.

        Args:
            - checkpoint (dict): Dictionary containing the current model state and other relevant information.
            - val_loss (float): Validation loss value.

        Returns:
            - None.
        """
        # Check if it's the first validation, or there is an improvement, or no improvement
        if self.best_score is None:
            self._handle_first_validation(checkpoint=checkpoint, val_loss=val_loss)
        elif val_loss > self.best_score + self.delta:
            self._handle_no_improvement()
        else:
            self._handle_improvement(checkpoint=checkpoint, val_loss=val_loss)

    def _handle_first_validation(self, checkpoint, val_loss):
        """
        Handle the case of the first validation.

        Args:
            - checkpoint (dict): Dictionary containing the current model state and other relevant information.
            - val_loss (float): Validation loss value.
        """
        self.best_score = val_loss
        self.save_checkpoint(checkpoint=checkpoint)

    def _handle_no_improvement(self):
        """
        Handle the case of no improvement in validation loss.
        """
        self.counter += 1
        print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            # if the counter exceeds patience, set the early_stop flag
            self.early_stop = True

    def _handle_improvement(self, checkpoint, val_loss):
        """
        Handle the case of an improvement in validation loss.

        Args:
            - checkpoint (dict): Dictionary containing the current model state and other relevant information.
            - val_loss (float): Validation loss value.
        """
        self.counter = 0
        self.best_score = val_loss
        self.save_checkpoint(checkpoint=checkpoint)

    def save_checkpoint(self, checkpoint):
        """
        Saves model when validation loss decreases.

        Args:
            - checkpoint (dict): Dictionary containing the current model state and other relevant information.
        """
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {self.best_score:.6f}). Saving model...")

        # remove the previous best model checkpoint if it exists
        if self.best_model_path is not None and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)

        # save the current best model's state_dict
        self.best_model_path = os.path.join(
            self.path, f"model-epoch={checkpoint['epoch'] + 1}-val_loss={self.best_score:.4f}.pt")

        torch.save(obj=checkpoint, f=self.best_model_path)

        self.val_loss_min = self.best_score
