import torch
import random
import numpy as np
import torch.nn as nn

def set_seeds(SEED=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

if __name__ == "__main__":
    print(f"\n --------------- nn.Linear() --------------- \n")

    # set seeds for reproducibility
    set_seeds()

    # generate a random input tensor x with shape (1, 2)
    x = torch.rand((1, 2))

    # create a linear layer l1 with input features=2 and output features=2
    set_seeds()
    l1 = nn.Linear(2, 2)

    # display the parameters of the linear layer
    for name, params in l1.named_parameters():
        print(name, params, params.shape)
    print()

    # calculate the result of the linear layer operation
    result_l1 = torch.matmul(l1.weight, x.T) + l1.bias.view(2, -1)

    # display the result and its shape
    print(f"\nResult: \n{result_l1}, {result_l1.shape}")

    print(f"\n --------------- nn.Conv2d() --------------- \n")

    # create a 2D convolutional layer l2 with input channels=2, output channels=2, and kernel size (1, 1)
    set_seeds()
    l2 = nn.Conv2d(2, 2, kernel_size=(1, 1), stride=1, padding=0)

    # display the parameters of the convolutional layer
    for name, params in l2.named_parameters():
        print(name, params, params.shape)

    # calculate the result of the convolutional layer operation
    result_l2 = torch.matmul(l2.weight.view(2, 2), x.T) + l2.bias.view(2, -1)

    # display the result and its shape
    print(f"\nResult: \n{result_l2}, {result_l2.shape}")

    print(f"\n --------------- nn.Parameter() --------------- \n")

    # create 
    set_seeds()
    w = nn.Parameter(data=torch.tensor([[0.5406, 0.5869], 
                                        [-0.1657, 0.6496]], dtype=torch.float32), requires_grad=True)    
    b = nn.Parameter(data=torch.tensor([-0.1549, 
                                        0.1427], dtype=torch.float32), requires_grad=True)
    
    print(f"{w}, {b}")
    
    result_l3 = torch.matmul(w, x.T) + b.view(2, -1)

    # display the result and its shape
    print(f"\nResult: \n{result_l3}, {result_l3.shape}\n")
