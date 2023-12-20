import torch
import torch.nn as nn


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        """
        Intersection over Union Loss.
        
        IoU = Area of Overlap / Area of Union
        IoU loss is modified to use for heatmaps.
        """
        self.eps = 1e-6

    def forward(self, input, target):
        """
        Calculate Intersection over Union (IoU) loss between predicted and target heatmaps.

        Args:
            - input (torch.Tensor): Predicted heatmap.
            - target (torch.Tensor): Target heatmap.

        Returns:
            - torch.Tensor: IoU loss.
        """
        input = input.view(-1)
        target = target.view(-1)

        intersection = (input * target).sum()

        union = (input.sum() + target.sum()) - intersection

        iou = (intersection / (union + self.eps))

        return 1 - iou


class IoULoss1(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        """
        Intersection over Union Loss.
        
        IoU = Area of Overlap / Area of Union
        IoU loss is modified to use for heatmaps.
        """
        from torchmetrics.classification import BinaryJaccardIndex
        self.criterion = BinaryJaccardIndex()

    def forward(self, input, target):
        """
        Calculate Intersection over Union (IoU) loss between predicted and target heatmaps.

        Args:
            - input (torch.Tensor): Predicted heatmap.
            - target (torch.Tensor): Target heatmap.

        Returns:
            - torch.Tensor: IoU loss.
        """
        iou = self.criterion(input, target)

        return 1 - iou


class ConvBlock(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding) -> None:
        super(ConvBlock, self).__init__()
        """
        Create a convolutional layer block with ReLU activation and batch normalization.

        Args:
            - in_channels (int): Number of input channels.
            - output_channels (int): Number of output channels.
            - kernel_size (int): Size of the convolutional kernel.
            - padding (int): Padding size for the convolutional layer.

        Returns:
            - None
        """
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=output_channels, 
                      kernel_size=kernel_size, 
                      padding=padding,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels, 
                      out_channels=output_channels, 
                      kernel_size=kernel_size, 
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the convolutional layer block.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        return self.conv_block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(UNet, self).__init__()
        """
        U-Net architecture for semantic segmentation.

        Args:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels.

        Returns:
            - None
        """
        self.down_1 = ConvBlock(in_channels=in_channels, output_channels=64, kernel_size=3, padding=1)
        self.down_2 = ConvBlock(in_channels=64, output_channels=128, kernel_size=3, padding=1)
        self.down_3 = ConvBlock(in_channels=128, output_channels=256, kernel_size=3, padding=1)
        self.down_4 = ConvBlock(in_channels=256, output_channels=512, kernel_size=3, padding=1)
        self.down_5 = ConvBlock(in_channels=512, output_channels=1024, kernel_size=3, padding=1)
        
        self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = ConvBlock(in_channels=1024, output_channels=512, kernel_size=3, padding=1)
        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = ConvBlock(in_channels=512, output_channels=256, kernel_size=3, padding=1)
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = ConvBlock(in_channels=256, output_channels=128, kernel_size=3, padding=1)
        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = ConvBlock(in_channels=128, output_channels=64, kernel_size=3, padding=1)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, padding=0)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            - x (torch.Tensor): Input image.

        Returns:
            - torch.Tensor: Predicted mask.
        """
        x1 = self.down_1(x)
        x2 = self.max_pool(x1)
        x3 = self.down_2(x2)
        x4 = self.max_pool(x3)
        x5 = self.down_3(x4)
        x6 = self.max_pool(x5)
        x7 = self.down_4(x6)
        x8 = self.max_pool(x7)
        x9 = self.down_5(x8)
        
        x = self.up_1(x9)
        x = self.up_conv_1(torch.cat([x, x7], 1))
        x = self.up_2(x)
        x = self.up_conv_2(torch.cat([x, x5], 1)) 
        x = self.up_3(x)
        x = self.up_conv_3(torch.cat([x, x3], 1))
        x = self.up_4(x)
        x = self.up_conv_4(torch.cat([x, x1], 1))

        x = self.output(x)
        
        return x
