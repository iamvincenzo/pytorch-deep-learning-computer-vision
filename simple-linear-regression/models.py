import torch
import torch.nn as nn
import torch.nn.init as init


class RawLinearRegrNet(nn.Module):
    """
    A simple linear regression model with learnable parameters.

    Args:
        nn (torch.nn.Module): PyTorch neural network module.

    Attributes:
        weights (torch.nn.Parameter): Learnable weights parameter.
        bias (torch.nn.Parameter): Learnable bias parameter.
    """
    def __init__(self):
        """
        Initializes the RawLinearRegrNet model with learnable parameters.
        """
        # call the constructor of the parent class (nn.Module)
        super(RawLinearRegrNet, self).__init__()

        # define learnable parameters: weights and bias
        # both initialized with random values
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # linear regression model: y = weights * x + bias
        return self.weights * x + self.bias


class LinearRegrNet(nn.Module):
    """
    Linear Regression model.

    Attributes:
        l1 (nn.Linear): Linear layer with input size 1 and output size 1.
    """
    def __init__(self):
        """
        Initializes the LinearRegrNet model with a linear layer.
        """
        # call the constructor of the parent class (nn.Module)
        super(LinearRegrNet, self).__init__()

        # define a linear layer: input size=1, output size=1
        self.l1 = nn.Linear(1, 1)

        # initialize weights using a method between 0 and 1
        init.uniform_(self.l1.weight, 0, 1)
        # initialize bias to zero
        init.constant_(self.l1.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 1).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 1).
        """
        # forward pass through the linear layer
        x = self.l1(x)

        return x
