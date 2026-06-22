import torch
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

EPOCHS = 1000

def set_seeds(SEED=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


class XORDataset(Dataset):
    def __init__(self, data):
        super(XORDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[:, :2][index], self.data[:, -1:][index]

    def __len__(self):
        return len(self.data)


class MLP(nn.Module):
    def __init__(self):
        """
        """
        super(MLP, self).__init__()
        self.w1 = nn.Parameter(data=torch.rand((2, 2), dtype=torch.float32), requires_grad=True)
        self.b1 = nn.Parameter(data=torch.rand((2, 1), dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(data=torch.rand((2), dtype=torch.float32), requires_grad=True)
        self.b2 = nn.Parameter(data=torch.rand((1), dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        """
        """
        x = self.w1 @ x.T + self.b1
        x = torch.max(torch.tensor(0.), x) # relu
        x = self.w2 @ x + self.b2
        x = 1 / (1 + torch.exp(-x)) # sigmoid

        return x

if __name__ == "__main__":
    set_seeds()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.tensor([[0., 0., 0.],
                         [0., 1., 1.],
                         [1., 0., 1.],
                         [1., 1., 0.]], dtype=torch.float32)
    
    train_dataset = XORDataset(data=data)

    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=1)

    model = MLP().to(device=dev)

    loss_fn = nn.BCELoss()

    optimizer = Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))

    train_losses = []
    avg_train_losses = []

    for epoch in range(EPOCHS):

        model.train()

        for idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device=dev), y.to(device=dev).squeeze()

            preds = model(X)

            loss = loss_fn(preds, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())
    
        epoch_loss = np.mean(train_losses)
        print(f"Epoch[{epoch + 1}/{EPOCHS}] loss: {epoch_loss}")
        avg_train_losses.append(np.mean(train_losses))

        train_losses = []

    # Plotting the average training losses
    epochs = np.arange(1, len(avg_train_losses) + 1)
    plt.plot(epochs, avg_train_losses, label="Average Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Training Loss over Epochs")
    plt.legend()
    plt.show()
