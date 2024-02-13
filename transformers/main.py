import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary

from model import Transformer


# paramateres
SEED = 42
LR = 0.001
EPOCHS = 1000
NUM_HEADS = 8
N_CLASSES = 1
DROPOUT = 0.5
EMBED_DIM = 512
PRINT_EVERY = 100
NUM_ENCODER_BLOCKS = 4

torch.manual_seed(42)


def get_probs(n_classes: int, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple:
    """
    Calculate class probabilities or binary predictions based on the model predictions.

    Args:
        - n_classes (int): The number of classes in the classification task.
        - y_pred (torch.Tensor): The model predictions tensor.
        - y_true (torch.Tensor): The true labels tensor.

    Returns:
        tuple(torch.Tensor, torch.Tensor): A tuple containing:
            - torch.Tensor: The predicted probabilities tensor for multi-class classification, or binary predictions tensor for binary classification.
            - torch.Tensor: The predicted class labels tensor.
    """
    if n_classes > 1:
        probs = torch.softmax(input=y_pred, dim=-1)
        preds = torch.argmax(input=probs, dim=-1, keepdim=True)
    else:
        probs = torch.sigmoid(input=y_pred)
        preds = torch.round(input=probs)
    
    return probs, preds


if __name__ == "__main__":
    # select the device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # define the model
    model = Transformer(num_encoder_blocks=NUM_ENCODER_BLOCKS, 
                        num_heads=NUM_HEADS, embed_dim=EMBED_DIM, n_classes=N_CLASSES, dropout=DROPOUT)
    model = model.to(device)

    # define training/validation data
    x_train = torch.randn(8, 1, 512)
    y_train = torch.rand(8, 1).round()

    # print the model summary
    summary(model=model, input_size=(x_train.shape[1:]), batch_size=x_train.shape[0], device=device)

    # define the loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # define the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    train_losses, val_losses = [], []

    # train the model
    model.train()
    for values in range(EPOCHS):
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        y_ = model(x_train)
                
        train_loss = loss_fn(y_, y_train)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if values % 10 == 9:
            train_losses.append(train_loss.item())

        # evaluate the model 
        model.eval()
        with torch.no_grad():
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            y_ = model(x_train)
            
            val_loss = loss_fn(y_, y_train)
            
            if values % 10 == 9:
                val_losses.append(val_loss)
        model.train()

        if (values + 1) % PRINT_EVERY == 0:
            print(f"epoch[{values + 1}/{EPOCHS}]: train-loss: {train_loss}, val-loss: {val_loss}")
    
            # plot train and validation losses
            epochs = range(1, len(train_losses) + 1)
            plt.plot(epochs, train_losses, color="blue", label="Train Loss")
            plt.plot(epochs, val_losses, color="red", label="Validation Loss")
            plt.yscale("log")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
