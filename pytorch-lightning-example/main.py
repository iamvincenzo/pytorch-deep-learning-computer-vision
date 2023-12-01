import torch
import torchvision
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler


# hyper-parameters
batch_size = 32
num_classes = 10
num_epochs = 100
hidden_size = 500
input_size = 28 * 28
learning_rate = 0.001


class LitModel(pl.LightningModule):
    """
    LightningModule for a simple neural network with one hidden layer.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initializes the LitModel model.

        Parameters:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            num_classes (int): Number of output classes.
        """
        super(LitModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.train_preds_list = []
        self.valid_preds_list = []
        self.train_labels_list = []
        self.valid_labels_list = []
        self.train_step_outputs = []
        self.valid_step_outputs = []

    def train_dataloader(self):
        """
        Provides the training DataLoader for the LightningModule.

        Returns:
            torch.utils.data.DataLoader: DataLoader for training data.
        """
        train_dataset = torchvision.datasets.MNIST(
            root="./pytorch-lightning-example/data",
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor()
            ]),
            download=True
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=15,
            persistent_workers=True,
            shuffle=True
        )

        return train_loader

    def val_dataloader(self):
        """
        Provides the validation DataLoader for the LightningModule.

        Returns:
            torch.utils.data.DataLoader: DataLoader for validation data.
        """
        valid_dataset = torchvision.datasets.MNIST(
            root="./pytorch-lightning-example/data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            download=True
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            num_workers=15,
            persistent_workers=True,
            shuffle=False
        )

        return valid_loader

    def configure_optimizers(self):
        """
        Configures the optimizer for the LightningModule.

        Returns:
            torch.optim.Optimizer: Optimizer for the model.
        """
        return torch.optim.Adam(params=self.parameters(), lr=learning_rate)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.l2(F.relu(self.l1(x)))

    def training_step(self, batch, batch_idx):
        """
        Training step for the LightningModule.

        Parameters:
            batch (tuple): Batch of input images and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: Dictionary containing the training loss and tensorboard logs.
        """
        images, labels = batch
        images = images.reshape(-1, 28 * 28)
        preds = self.forward(images)
        loss = F.cross_entropy(input=preds, target=labels)
        
        self.train_preds_list.append(preds)
        self.train_labels_list.append(labels)
        self.train_step_outputs.append(loss)

        # log generated images to TensorBoard
        # accuracy = self.accuracy(preds, labels)
        # f1_score = self.f1_score(preds, labels)
        # self.log_dict(
        #     {"train_loss": loss, "train_accuracy": accuracy, "train_f1_score": f1_score},
        #     on_step=False, on_epoch=True, prog_bar=True
        # )

        if batch_idx % 100 == 0:
            images = images[:8]
            grid = torchvision.utils.make_grid(images.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("generated_images", grid, self.global_step)

        return loss

    def on_train_epoch_end(self):
        """
        Actions to perform at the end of each training epoch.
        """
        preds = torch.cat(self.train_preds_list, dim=0)
        labels = torch.cat(self.train_labels_list, dim=0)
        epoch_mean = torch.stack(self.train_step_outputs).mean()
        
        accuracy = self.accuracy(preds, labels)
        f1_score = self.f1_score(preds, labels)
        self.log_dict({"train_loss": epoch_mean, "train_accuracy": accuracy, "train_f1_score": f1_score},
                      on_step=False, on_epoch=True, prog_bar=True)
        
        # free up the memory
        self.train_preds_list.clear()
        self.train_labels_list.clear()
        self.train_step_outputs.clear()        

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the LightningModule.

        Parameters:
            batch (tuple): Batch of input images and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: Dictionary containing the validation loss and tensorboard logs.
        """
        images, labels = batch
        images = images.reshape(-1, 28 * 28)
        preds = self.forward(images)
        loss = F.cross_entropy(input=preds, target=labels)

        self.valid_preds_list.append(preds)
        self.valid_labels_list.append(labels)
        self.valid_step_outputs.append(loss)

        accuracy = self.accuracy(preds, labels)
        f1_score = self.f1_score(preds, labels)
        self.log_dict(
            {"valid_loss": loss, "valid_accuracy": accuracy, "valid_f1_score": f1_score},
            on_step=False, on_epoch=True, prog_bar=True
        )

        return loss


# check if the script is being run as the main program
if __name__ == "__main__":
    logger = TensorBoardLogger(save_dir="./pytorch-lightning-example/tb_logs", name="mnist_model_v1")

    # profiler = PyTorchProfiler(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./pytorch-lightning-example/tb_logs/profiler0"),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # )

    early_stop_callback = EarlyStopping(monitor="valid_loss", patience=5, verbose=True, mode="min")

    trainer = Trainer(fast_dev_run=False, logger=logger, min_epochs=1, # profiler="simple",
                      max_epochs=num_epochs, accelerator="cpu", callbacks=[early_stop_callback])

    model = LitModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

    trainer.fit(model=model)
