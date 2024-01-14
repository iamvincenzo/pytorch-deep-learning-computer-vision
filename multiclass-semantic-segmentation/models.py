import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.models.segmentation import deeplabv3_resnet101


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


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 3) -> None:
        """
        U-Net architecture for semantic segmentation.

        Args:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels.

        Returns:
            - None.
        """
        super(UNet, self).__init__()
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
        
        self.output = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, padding=0)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# def prepare_model(num_classes=3):
#     model = deeplabv3_resnet101(weights="DEFAULT")
    
#     model.classifier[4] = nn.Conv2d(256, num_classes, 1)
#     model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    
#     return model


# error in loss.backward() because torch.argmax() detach network output from the graph because it is not differentiable
# https://discuss.pytorch.org/t/element-0-of-tensors-does-not-require-grad-and-does-not-have-a-grad-fn-how-can-i-fix-this/140756
# 
# class IOULoss(nn.Module):
#     def __init__(self, num_classes: int, average: str = "macro", weight: torch.Tensor = None) -> None:
#         """
#         Intersection over Union (IOU) loss for multiclass semantic segmentation.

#         Args:
#             - num_classes (int): Number of classes in the segmentation task.
#             - average (str, optional): Specifies whether to compute the average loss across the batch.
#                 Possible values are "none" or "macro". Default is "macro".
#             - weight (Tensor, optional): A tensor of weights for each class. If None, all classes have equal weight.
#                 Default is None.
        
#         Returns:
#             - None.
#         """
#         super(IOULoss, self).__init__()
#         self.weight = weight
#         self.average = average
#         self.num_classes = num_classes
#         self.metric = MulticlassJaccardIndex(average=average, num_classes=num_classes)
#         if self.average == "none":
#             assert self.weight is not None, "Weight should be provided when average='none'."
#             assert self.weight.dtype == torch.float32, "Input tensor 'weight' must be of type float32."

#     def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the IOU loss for multiclass semantic segmentation.

#         Args:
#             - logits (Tensor): Predicted segmentation masks (logits).
#             - target (Tensor): Ground truth segmentation masks (class indices).

#         Returns:
#             - Tensor: IOU loss.
#         """
#         # since we are using CrossEntropyLoss: logits --> probabilities --> labels
#         probs = torch.softmax(logits, dim=1)
#         predicted = torch.argmax(probs, dim=1)

#         assert predicted.dtype == torch.int64, "Input tensor 'predicted' must be of type int64."
#         assert target.dtype == torch.int64, "Input tensor 'target' must be of type int64."

#         iou = self.metric(predicted, target)

#         iou_loss = 1 - iou

#         if self.average == "none":
#             iou_loss = (iou_loss * self.weight).mean()

#         return iou_loss