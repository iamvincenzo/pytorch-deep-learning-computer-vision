import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary

from model import Transformer


torch.manual_seed(42)

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = Transformer(num_encoder_blocks=6, num_heads=8, embed_dim=512, n_classes=1)
    model = model.to(device)

    x = torch.randn(8, 1, 512)
    y = torch.rand(8, 1, 1).round()

    summary(model=model, input_size=(x.shape[1:]), batch_size=x.shape[0], device="cpu")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    train_losses, val_losses = [], []

    model.train()

    for values in range(1000):
        y_ = model(x)

        train_loss = loss_fn(y_, y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if values % 10 == 9:
            train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            y_ = model(x)
            val_loss = loss_fn(y_, y)
            if values % 10 == 9:
                val_losses.append(val_loss)
        model.train()

        if values % 100 == 99:
            print(f"epoch[{values}/500]: train-loss: {train_loss}, val-loss: {val_loss}")
            plt.plot(train_losses, color='blue', label='Train Loss')
            plt.plot(val_losses, color='red', label='Validation Loss')
            plt.yscale('log')  # setting log scale for y-axis
            plt.legend()
            plt.show()

