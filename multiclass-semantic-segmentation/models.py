import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.models.segmentation import deeplabv3_resnet101


class IOULoss(nn.Module):
    def __init__(self, num_classes, average="none", weight=None):
        """
        Intersection over Union (IOU) loss for multiclass semantic segmentation.

        Args:
            - num_classes (int): Number of classes in the segmentation task.
            - average (str, optional): Specifies whether to compute the average loss across the batch.
                Possible values are "none", "micro", "macro", "weighted", or None. Default is "none".
            - weight (Tensor, optional): A tensor of weights for each class. If None, all classes have equal weight.
                Default is None.
        """
        super(IOULoss, self).__init__()
        self.metric = MulticlassJaccardIndex(num_classes=num_classes, average=average)
        self.num_classes = num_classes
        self.average = average
        self.weight = weight

    def forward(self, predicted, target):
        """
        Compute the IOU loss for multiclass semantic segmentation.

        Args:
            - predicted (Tensor): Predicted segmentation masks (logits).
            - target (Tensor): Ground truth segmentation masks (class indices).

        Returns:
            - Tensor: IOU loss.
        """
        iou = self.metric(predicted, target)

        iou_loss = 1 - iou

        if self.average == "none":
            iou_loss = (iou_loss * self.weight).mean()

        return iou_loss

def conv_layer(input_channels, output_channels):
    """
    Create a convolutional layer block with ReLU activation and batch normalization.

    Args:
        - input_channels (int): Number of input channels.
        - output_channels (int): Number of output channels.

    Returns:
        - nn.Sequential: Convolutional layer block.
    """
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU()
    )

    return conv

class UNet(nn.Module):
    def __init__(self, n_classes):
        """
        U-Net architecture for semantic segmentation.
        """
        super(UNet, self).__init__()
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_1 = conv_layer(3, 64)
        self.down_2 = conv_layer(64, 128)
        self.down_3 = conv_layer(128, 256)
        self.down_4 = conv_layer(256, 512)
        self.down_5 = conv_layer(512, 1024)
        
        self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = conv_layer(1024, 512)
        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = conv_layer(512, 256)
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = conv_layer(256, 128)
        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = conv_layer(128, 64)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, padding=0)
                
    def forward(self, img):
        """
        Forward pass of the U-Net model.

        Args:
            - img (torch.Tensor): Input image.

        Returns:
            - torch.Tensor: Predicted mask.
        """
        x1 = self.down_1(img)
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