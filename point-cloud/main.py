import torch
import torch.nn as nn
import torch.nn.functional as F


def test_pointcloud():
    # sample 3D point cloud with three features (x, y, z)
    # batch size: 1, Number of points: 5, Number of features: 3
    point_cloud = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ]
        ]
    )

    torch.manual_seed(42)

    print(f"Point-cloud shape before: {point_cloud.shape}")

    print(point_cloud)

    # reshape the input to (batch_size, features, n-points) for Conv1D
    point_cloud = point_cloud.permute(0, 2, 1)

    print(f"Point-cloud shape after: {point_cloud.shape}")

    # define a simple 1D convolutional layer
    conv1d_layer = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=1)

    # apply the convolution operation
    output = conv1d_layer(point_cloud)

    # print the input, convolution operation, and output
    print("Input (3D point cloud):")
    print(point_cloud)

    print("\nConv1D Layer:")
    print(conv1d_layer.weight.data)
    print(conv1d_layer.bias.data)

    print("\nOutput after Conv1D:")
    print(output)

    # print manual calculations for the first output values
    print((1 * 0.4414) + (2 * 0.4792) + (3 * -0.1353) - 0.2811)  # 1st kernel
    print((4 * 0.4414) + (5 * 0.4792) + (6 * -0.1353) - 0.2811)  # 1st kernel
    print((7 * 0.4414) + (8 * 0.4792) + (9 * -0.1353) - 0.2811)  # 1st kernel
    print((13 * 0.5304) + (14 * -0.1265) + (15 * 0.1165) + 0.3391)  # 2nd kernel


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

    def forward(self, x):
        pass

class Solver(object):
    def __init__(self):
        pass

    def train_net(self):
        pass

    def valid_net(self):
        pass

if __name__ == "__main__":
    # test_pointcloud()
    pass
