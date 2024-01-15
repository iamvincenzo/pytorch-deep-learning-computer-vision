# source: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, output_channels: int, kernel_size: int, padding: int) -> None:
        """
        Create a convolutional layer block with ReLU activation and batch normalization.

        Args:
            - in_channels (int): Number of input channels.
            - output_channels (int): Number of output channels.
            - kernel_size (int): Size of the convolutional kernel.
            - padding (int): Padding size for the convolutional layer.

        Returns:
            - None.
        """
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=output_channels, 
                      kernel_size=kernel_size, 
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels, 
                      out_channels=output_channels, 
                      kernel_size=kernel_size, 
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional layer block.

        Args:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor.
        """
        return self.conv_block(x)


class AttentionGate(nn.Module):
    def __init__(self, g_in_channels: int, x_in_channels: int, out_channels: int) -> None:
        """
        Initialize the AttentionGate module.

        Parameters:
            - g_in_channels (int): Number of input channels for the 'g' tensor.
            - x_in_channels (int): Number of input channels for the 'x' tensor.

        Returns:
            - None.
        """
        super(AttentionGate, self).__init__()
        # define a 1×1 convolutional layer for the 'g' tensor
        self.g_block = nn.Sequential(
            nn.Conv2d(in_channels=g_in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        # define a 1×1 convolutional layer for the 'x' tensor
        self.x_block = nn.Sequential(
            nn.Conv2d(in_channels=x_in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        # define a 1×1 convolutional layer for the 'x' tensor to compute 'psi'
        self.psi_block = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # self.resampler_conv = nn.Sequential(
        #     ConvBlock(in_channels=x_in_channels, output_channels=x_in_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(x_in_channels)
        # )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionGate module.

        Parameters:
            - g (torch.Tensor): Tensor representing the 'g' input.
            - x (torch.Tensor): Tensor representing the 'x' input.

        Returns:
            - torch.Tensor: Output tensor after attention gating.
        """
        # apply 1x1 convolution to 'g'
        w_g = self.g_block(g)
        # apply 1x1 convolution to 'x' with strided convolution
        w_x = self.x_block(x)
        # element-wise addition of the two convolutional outputs
        sigma1 = torch.add(w_g, w_x)
        # apply ReLU activation
        sum = torch.relu(sigma1)
        # apply 1x1 convolution to 'sum'
        psi = self.psi_block(sum)
        # element-wise multiplication of the soft-attention weights and the original 'x'
        y = torch.mul(x, psi)

        # # further process the upsampled tensor with a convolutional block
        # x_caret = self.resampler_conv(y)
        # return x_caret

        return y


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 3) -> None:
        """
        Attention U-Net architecture for semantic segmentation.

        Args:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels (classes).

        Returns:
            - None.
        """
        super(AttentionUNet, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_1 = ConvBlock(in_channels=in_channels, output_channels=64, kernel_size=3, padding=1)
        self.down_2 = ConvBlock(in_channels=64, output_channels=128, kernel_size=3, padding=1)
        self.down_3 = ConvBlock(in_channels=128, output_channels=256, kernel_size=3, padding=1)
        self.down_4 = ConvBlock(in_channels=256, output_channels=512, kernel_size=3, padding=1)
        self.down_5 = ConvBlock(in_channels=512, output_channels=1024, kernel_size=3, padding=1)
        
        self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.attention_block_1 = AttentionGate(g_in_channels=512, x_in_channels=512, out_channels=256)
        self.up_conv_1 = ConvBlock(in_channels=1024, output_channels=512, kernel_size=3, padding=1)

        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.attention_block_2 = AttentionGate(g_in_channels=256, x_in_channels=256, out_channels=128)
        self.up_conv_2 = ConvBlock(in_channels=512, output_channels=256, kernel_size=3, padding=1)

        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.attention_block_3 = AttentionGate(g_in_channels=128, x_in_channels=128, out_channels=64)
        self.up_conv_3 = ConvBlock(in_channels=256, output_channels=128, kernel_size=3, padding=1)

        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.attention_block_4 = AttentionGate(g_in_channels=64, x_in_channels=64, out_channels=32)
        self.up_conv_4 = ConvBlock(in_channels=128, output_channels=64, kernel_size=3, padding=1)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention U-Net model.

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
        
        g = self.up_1(x9)
        x7_caret = self.attention_block_1(g, x7)
        g = self.up_conv_1(torch.cat([x7_caret, g], 1))

        g = self.up_2(g)
        x5_caret = self.attention_block_2(g, x5)
        g = self.up_conv_2(torch.cat([x5_caret, g], 1))

        g = self.up_3(g)
        x3_caret = self.attention_block_3(g, x3)
        g = self.up_conv_3(torch.cat([x3_caret, g], 1))

        g = self.up_4(g)
        x1_caret = self.attention_block_4(g, x1)
        x = self.up_conv_4(torch.cat([x1_caret, g], 1))  

        x = self.output(x)
        
        return x


if __name__ == "__main__":
    x = torch.rand(1, 3, 256, 256)

    model = AttentionUNet(in_channels=3, n_classes=3)

    print(model(x).shape)
