import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    ResidualBlock: Class representing a basic building block of the ResNet architecture.
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the convolutional layers. Default is 1.

    Attributes:
        conv_res1, conv_res2 (nn.Conv2d): Convolutional layers.
        bn_res1, bn_res2 (nn.BatchNorm2d): Batch normalization layers.
        relu (nn.ReLU): ReLU activation function.
        downsample (nn.Sequential or None): Downsample layer if stride is not 1, else None.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # first convolutional block
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                   kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_res1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # second convolutional block
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_res2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

        # downsample layer for shortcut connection if stride is not 1
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        else:
            self.downsample = None

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # save the input for the shortcut connection
        identity = x

        # first convolutional block
        x = self.conv_res1(x)
        x = self.bn_res1(x)
        x = self.relu(x)

        # second convolutional block
        x = self.conv_res2(x)
        x = self.bn_res2(x)

        # apply shortcut connection if downsample is not None
        if self.downsample is not None:
            identity = self.downsample(identity)

        # element-wise addition of the shortcut connection
        x += identity

        return x


class ResNet18(nn.Module):
    """
    ResNet18: Implementation of the ResNet-18 architecture for image classification.

    Args:
        input_size (tuple): Tuple representing the input size (in_channels, height, width).
        output_size (int): Number of output classes.
        weights_init (bool): Flag to perform weight initialization. Default is True.

    Attributes:
        conv1, bn1, relu1, max_pool (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d): Starting block of ResNet.
        conv2_1, conv2_2, conv3_1, conv3_2, conv4_1, conv4_2, conv5_1, conv5_2 (ResidualBlock): Residual blocks.
        adpt_avg_pool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        fc (nn.Linear): Fully connected layer for the final output.
    """
    def __init__(self, input_size, output_size=16, weights_init=True):
        """
        Initializes the ResNet18 model.

        Args:
            input_size (tuple): Tuple representing the input size (in_channels, height, width).
            output_size (int): Number of output classes.
            weights_init (bool): Flag to perform weight initialization. Default is True.
        """
        super(ResNet18, self).__init__()
        (in_channels, h, w) = input_size

        # starting block of ResNet
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # residual blocks
        self.conv2_1 = ResidualBlock(in_channels=64, out_channels=64, stride=1)
        self.conv2_2 = ResidualBlock(in_channels=64, out_channels=64, stride=1)
        self.conv3_1 = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        self.conv3_2 = ResidualBlock(in_channels=128, out_channels=128, stride=1)
        self.conv4_1 = ResidualBlock(in_channels=128, out_channels=256, stride=2)
        self.conv4_2 = ResidualBlock(in_channels=256, out_channels=256, stride=1)
        self.conv5_1 = ResidualBlock(in_channels=256, out_channels=512, stride=2)
        self.conv5_2 = ResidualBlock(in_channels=512, out_channels=512, stride=1)

        # final part
        self.adpt_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_size, bias=True)

        # perform weights initialization if required
        if weights_init:
            self.weights_initialization()

    def weights_initialization(self):
        """
        Performs weight initialization for the ResNet18 model.
        """
        print(f"\nPerforming weights initialization...")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.trunc_normal_(m.weight, mean=0, std=2**(-0.5))
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.trunc_normal_(m.weight, mean=0, std=2**(-0.5))
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Defines the forward pass through the ResNet18 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.max_pool(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.adpt_avg_pool(x)
        x = x.reshape(x.size(0), -1)
        logits = self.fc(x)

        return logits


class MultiLabelImageClassifier(nn.Module):
    """
    MultiLabelImageClassifier is a convolutional neural network (CNN) designed for
    multi-label image classification using PyTorch.

    Args:
        num_classes (int): Number of output classes for the final fully connected layer.
                           Default is 16.

    Attributes:
        conv1, conv2, conv3, conv4 (nn.Sequential): Convolutional layers with batch normalization
            and ReLU activation.
        maxpool (nn.MaxPool2d): Max pooling layer.
        fc1, fc2, fc3 (nn.Sequential): Fully connected layers with batch normalization and ReLU activation.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
    """
    def __init__(self, num_classes=16):
        """
        Initializes the MultiLabelImageClassifier.

        Args:
            num_classes (int): Number of output classes for the final fully connected layer.
                               Default is 16.
        """
        # initialize the MultiLabelImageClassifier class, inheriting from nn.Module
        super(MultiLabelImageClassifier, self).__init__()

        # define convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # define pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # define fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(512, num_classes)

        # define dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Defines the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing the image data.

        Returns:
            torch.Tensor: Output logits representing the class scores.
        """
        # forward pass through the network
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.maxpool(x)

        # flatten the input for the fully connected layers
        x = x.view(x.size(0), -1)

        # apply fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # final output layer
        logits = self.fc3(x)

        return logits
