import torch
import numpy as np
from tqdm import tqdm
from torchmetrics import ROC
from sklearn.metrics import auc
from timeit import default_timer as timer
from torchmetrics import PrecisionRecallCurve
from sklearn.metrics import cohen_kappa_score

from plotting_utils import plot_roc
from plotting_utils import plot_prc
from early_stopping import EarlyStopping
from plotting_utils import plot_confusion_matrix


class Solver(object):
    def __init__(self, epochs, writer, train_loader, test_loader, device, model, optimizer, criterion, patience):
        """
        A class to handle training and validation of a PyTorch neural network.

        Args:
            epochs (int): Number of training epochs.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            test_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            device (torch.device): Device on which to perform training and validation.
            model (torch.nn.Module): Neural network model.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
            criterion (torch.nn.Module): Loss function.
            patience (int): Number of epochs to wait for improvement before early stopping.
        """
        self.epochs = epochs
        self.writer = writer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.patience = patience

        self.model_reslts = {
            "test_name" : "",
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

        train_time_start = timer()
        
        self.model.train()

        for epoch in range(self.epochs):
            print(f"\nTraining iteration | Epoch[{epoch + 1}/{self.epochs}]")

            predictions = torch.tensor([], device=self.device)
            targets = torch.tensor([], device=self.device)

            # use tqdm for a progress bar during training
            loop = tqdm(iterable=enumerate(self.train_loader),
                        total=len(self.train_loader),
                        leave=True)

            for batch_idx, (x_train, y_train) in loop:
                # move data and labels to the specified device
                x_train = x_train.to(self.device)
                y_train = y_train.unsqueeze(1).to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                logits = self.model(x_train)

                # calculate the loss
                loss = self.criterion(logits, y_train)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                self.optimizer.step()

                # record training loss
                train_losses.append(loss.item())

                # since we are using BCEWithLogitsLoss
                # logits --> probabilities --> labels
                probs = torch.sigmoid(logits)
                y_pred = torch.round(probs)

                # record predictions and true labels
                predictions = torch.cat([predictions, y_pred], dim=0)
                targets = torch.cat([targets, y_train], dim=0)

            self.compute_metrics(epoch=epoch, 
                                 batch_idx=batch_idx, 
                                 predictions=predictions,
                                 targets=targets,
                                 train=True)

            # validate the model on the validation set
            self.valid_net(epoch=epoch, valid_losses=valid_losses)

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.epochs}] | train-loss: {train_loss:.4f} | "
                  f"validation-loss: {valid_loss:.4f}")
            
            self.writer.add_scalar("training-loss", train_loss, 
                                   epoch * len(self.train_loader) + batch_idx)
            self.writer.add_scalar("validation-loss", valid_loss, 
                                   epoch * len(self.test_loader) + batch_idx)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early stopping checks for improvement in validation loss
            early_stopping(epoch, valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping...")
                break
        
        print("\nTraining model Done...")

        train_time_end = timer()

        self.model_reslts["tot_epochs"] = epoch + 1
        self.model_reslts["train_time"] = train_time_end - train_time_start

        # write all remaining data in the buffer
        self.writer.flush()
        # free up system resources used by the writer
        self.writer.close() 

    def valid_net(self, epoch, valid_losses):
        """
        Validates the neural network on the specified DataLoader for validation data.

        Records validation losses in the provided list.

        Args:
            valid_losses (list): List to record validation losses.
        """
        print(f"\nStarting validation...\n")

        predictions = torch.tensor([], device=self.device)
        targets = torch.tensor([], device=self.device)

        self.model.eval()

        # use tqdm for a progress bar during validation
        with torch.inference_mode():
            loop = tqdm(iterable=enumerate(self.test_loader), 
                        total=len(self.test_loader), 
                        leave=True)

            for _, (x_valid, y_valid) in loop:
                # move data and labels to the specified device
                x_valid = x_valid.to(self.device)
                y_valid = y_valid.unsqueeze(1).to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                logits = self.model(x_valid)

                # calculate the loss
                loss = self.criterion(logits, y_valid)

                # record validation loss
                valid_losses.append(loss.item())

                # since we are using BCEWithLogitsLoss
                # logits --> probabilities --> labels
                probs = torch.sigmoid(logits)
                y_pred = torch.round(probs)

                # record predictions and true labels
                predictions = torch.cat([predictions, y_pred], dim=0)
                targets = torch.cat([targets, y_valid], dim=0)

            self.compute_metrics(epoch=epoch,
                                 batch_idx=1,
                                 predictions=predictions,
                                 targets=targets,
                                 train=False)
            
        # set the model back to training mode
        self.model.train()

    def eval_model(self):
        """
        Evaluates a given model on a given dataset.

        Args:
            model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
            data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
            loss_fn (torch.nn.Module): The loss function of model.
            accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
            device (str, optional): Target device to compute on. Defaults to device.

        Returns:
            (dict): Results of model making predictions on data_loader.
        """    
        print(f"\nStarting final evaluation...\n")

        predictions = torch.tensor([], device=self.device)
        targets = torch.tensor([], device=self.device)
        eval_losses = [] 

        self.model.eval()

        # use tqdm for a progress bar during validation
        with torch.inference_mode():
            loop = tqdm(iterable=enumerate(self.test_loader), 
                        total=len(self.test_loader), 
                        leave=True)

            for _, (x_valid, y_valid) in loop:
                # move data and labels to the specified device
                x_valid = x_valid.to(self.device)
                y_valid = y_valid.unsqueeze(1).to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                logits = self.model(x_valid)

                # calculate the loss
                loss = self.criterion(logits, y_valid)

                # record validation loss
                eval_losses.append(loss.item())

                # since we are using BCEWithLogitsLoss
                # logits --> probabilities --> labels
                probs = torch.sigmoid(logits)
                y_pred = torch.round(probs)

                # record predictions and true labels
                predictions = torch.cat([predictions, y_pred], dim=0)
                targets = torch.cat([targets, y_valid], dim=0)

            eval_loss = np.average(eval_losses)

            targets = targets.squeeze()
            predictions = predictions.squeeze()
            accuracy = torch.sum(predictions == targets).item() / targets.size(0)

            self.model_reslts["model_acc"] = accuracy
            self.model_reslts["model_loss"] = eval_loss

            return self.model_reslts

    def compute_metrics(self, epoch, batch_idx, predictions, targets, train):
        # compute accuracy
        # accuracy = torch.sum(predictions == targets).item() / (targets.size(0) * targets.size(1))

        true_positives = torch.sum((predictions == 1) & (targets == 1)).item()
        true_negatives = torch.sum((predictions == 0) & (targets == 0)).item()
        false_positives = torch.sum((predictions == 1) & (targets == 0)).item()
        false_negatives = torch.sum((predictions == 0) & (targets == 1)).item()

        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives 
                                                        + false_positives + false_negatives + 1e-20)
        precision = true_positives / (true_positives + false_positives + 1e-20)
        recall = true_positives / (true_positives + false_negatives + 1e-20)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-20)

        cohen_kappa = cohen_kappa_score(y1=targets.detach().cpu().numpy(), 
                                        y2=predictions.detach().cpu().numpy(), 
                                        weights='quadratic')
        
        # create the confusion matrix
        confusion_matrix = torch.tensor([[true_negatives, false_positives], 
                                         [false_negatives, true_positives]])
        
        # # normalize the confusion matrix
        # row_sums = confusion_matrix.sum(dim=1, keepdim=True)
        # normalized_confusion_matrix = confusion_matrix / row_sums


        # print accuracy based on training or validation
        metric_type = "train" if train else "valid"
        print(f"\n{metric_type} accuracy: {accuracy:.3f} | "
              f"precision: {precision:.3f} | recall: {recall:.3f} | "
              f"f1_score: {f1_score:.3f} | cohen-kappa-score: {cohen_kappa:.3f}")
        
        print(f"\nConfusion_matrix: \n{confusion_matrix}")
        # print(f"\nNormalized Confusion_matrix: \n{normalized_confusion_matrix}")
        
        self.writer.add_scalar(f"{metric_type}-accuracy", accuracy, epoch * len(self.train_loader) + batch_idx)
        self.writer.add_scalar(f"{metric_type}-precision", precision, epoch * len(self.train_loader) + batch_idx)
        self.writer.add_scalar(f"{metric_type}-recall", recall, epoch * len(self.train_loader) + batch_idx)
        self.writer.add_scalar(f"{metric_type}-f1_score", f1_score, epoch * len(self.train_loader) + batch_idx)

        fig = plot_confusion_matrix(tn=true_negatives, fp=false_positives, 
                                    fn=false_negatives, tp=true_positives)
        self.writer.add_figure(f"{metric_type}-confusion_matrix", fig, global_step=epoch * len(self.train_loader) + batch_idx)   

        roc = ROC(task="binary")
        fpr, tpr, thresholds = roc(predictions.detach().cpu(), 
                                   targets.type(torch.LongTensor).detach().cpu())
        roc_auc = auc(fpr, tpr)
        fig = plot_roc(fpr=fpr, tpr=tpr, auc=roc_auc)
        self.writer.add_figure(f"{metric_type}-ROC", fig, global_step=epoch * len(self.train_loader) + batch_idx)

        pr_curve = PrecisionRecallCurve(task="binary", thresholds=2)
        pr, rc, thresholds = pr_curve(predictions.detach().cpu(), 
                                      targets.type(torch.LongTensor).detach().cpu())
        fig = plot_prc(precision=pr, recall=rc)
        self.writer.add_figure(f"{metric_type}-PRC", fig, global_step=epoch * len(self.train_loader) + batch_idx)
