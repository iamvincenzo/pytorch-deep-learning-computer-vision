import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.losses import JaccardLoss
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.models.segmentation import deeplabv3_resnet101


# not works
class IOULoss(nn.Module):
    def __init__(self, num_classes: int, average: str = "macro", weight: torch.Tensor = None) -> None:
        """
        Intersection over Union (IOU) loss for multiclass semantic segmentation.

        Args:
            - num_classes (int): Number of classes in the segmentation task.
            - average (str, optional): Specifies whether to compute the average loss across the batch.
                Possible values are "none" or "macro". Default is "macro".
            - weight (Tensor, optional): A tensor of weights for each class. If None, all classes have equal weight.
                Default is None.
        
        Returns:
            - None.
        """
        super(IOULoss, self).__init__()
        self.weight = weight
        self.average = average
        self.num_classes = num_classes
        self.metric = MulticlassJaccardIndex(average=average, num_classes=num_classes)
        if self.average == "none":
            assert self.weight is not None, "Weight should be provided when average='none'."
            assert self.weight.dtype == torch.float32, "Input tensor 'weight' must be of type float32."

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the IOU loss for multiclass semantic segmentation.

        Args:
            - logits (Tensor): Predicted segmentation masks (logits).
            - target (Tensor): Ground truth segmentation masks (class indices).

        Returns:
            - Tensor: IOU loss.
        """
        # error in loss.backward() because torch.argmax() detach network output from the graph because it is not differentiable
        # https://discuss.pytorch.org/t/element-0-of-tensors-does-not-require-grad-and-does-not-have-a-grad-fn-how-can-i-fix-this/140756
        # since we are using CrossEntropyLoss: logits --> probabilities --> labels
        probs = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1)

        assert predicted.dtype == torch.int64, "Input tensor 'predicted' must be of type int64."
        assert target.dtype == torch.int64, "Input tensor 'target' must be of type int64."

        iou = self.metric(predicted, target)

        iou_loss = 1 - iou

        if self.average == "none":
            iou_loss = (iou_loss * self.weight).mean()

        return iou_loss


class CustomLoss(nn.Module):
    def __init__(self, n_classes: int, device: str, ce_weights: torch.Tensor, avg: bool = True, W_CE: float = 1., W_IoU: float = 1., W_Dice: float = 1.) -> None:
        """
        Custom loss function for multiclass semantic segmentation. A combination of 
        Cross entropy loss, Iou loss and Dice loss.

        Args:
            - n_classes (int): Number of classes in the segmentation task.
            - device (str): Device on which the model and loss should be placed (e.g., 'cuda' or 'cpu').
            - ce_weights (Tensor): Weights for class balancing in CrossEntropyLoss.
            - W_CE (float): Weight for the CrossEntropyLoss component.
            - W_IoU (float): Weight for the Intersection over Union (IoU) loss component.
            - W_Dice (float): Weight for the Dice loss component.

        Returns:
            - None.
        """
        super(CustomLoss, self).__init__()
        self.avg = avg
        self.W_CE = W_CE
        self.W_IoU = W_IoU
        self.W_Dice = W_Dice
        classes = np.arange(n_classes).tolist()
        self.ce_loss = CrossEntropyLoss(weight=ce_weights.to(device)).to(device)
        self.dice_loss = DiceLoss(mode="multiclass", classes=classes, from_logits=True).to(device)
        self.iou_loss = JaccardLoss(mode="multiclass", classes=classes, from_logits=True).to(device)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the custom loss for multiclass semantic segmentation.

        Args:
            - logits (Tensor): Predicted segmentation masks (logits).
            - target (Tensor): Ground truth segmentation masks (class indices).

        Returns:
            - torch.Tensor: A tensor containing losses.
        """
        assert logits.dtype == torch.float32, "Input tensor 'logits' must be of type float32."
        assert target.dtype == torch.int64, "Input tensor 'target' must be of type int64."

        loss_dict = {"ce_loss": self.W_CE * self.ce_loss(logits, target),
                     "iou_loss": self.W_IoU * self.iou_loss(logits, target),
                     "dice_loss": self.W_Dice * self.dice_loss(logits, target)}
        
        # print(loss_dict)
        
        losses = sum(loss for loss in loss_dict.values())

        # print(losses)

        if self.avg:
            losses /= len(loss_dict)

        return losses


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
    def __init__(self, g_in_channels: int, x_in_channels: int) -> None:
        """
        Initialize the AttentionGate module.

        Parameters:
            - g_in_channels (int): Number of input channels for the 'g' tensor.
            - x_in_channels (int): Number of input channels for the 'x' tensor.

        Returns:
            - None.
        """
        super(AttentionGate, self).__init__()
        self.g_block = nn.Sequential(
            nn.Conv2d(in_channels=g_in_channels, out_channels=x_in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(x_in_channels)
        )
        self.x_block = nn.Sequential(
            nn.Conv2d(in_channels=x_in_channels, out_channels=x_in_channels, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(x_in_channels)
        )
        self.psi_block = nn.Sequential(
            nn.Conv2d(in_channels=x_in_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.resampler = nn.ConvTranspose2d(in_channels=1, out_channels=x_in_channels, kernel_size=2, stride=2)
        # self.resampler = nn.Upsample(scale_factor=2, mode="nearest")
        self.resampler_conv = nn.Sequential(
            ConvBlock(in_channels=x_in_channels, output_channels=x_in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(x_in_channels)
        )

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
        # apply sigmoid activation
        sigma2 = torch.sigmoid(psi)
        # upsample 'x' using transposed convolution
        alpha = self.resampler(sigma2)
        # element-wise multiplication of the soft-attention weights and the original 'x'
        y = torch.mul(alpha, x)
        # further process the upsampled tensor with a convolutional block
        x_caret = self.resampler_conv(y)

        return x_caret


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 3) -> None:
        """
        U-Net architecture for semantic segmentation.

        Args:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels (classes).

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
        x = self.up_conv_1(torch.cat([x7, x], dim=1))
        x = self.up_2(x)
        x = self.up_conv_2(torch.cat([x5, x], dim=1)) 
        x = self.up_3(x)
        x = self.up_conv_3(torch.cat([x3, x], dim=1))
        x = self.up_4(x)
        x = self.up_conv_4(torch.cat([x1, x], dim=1))

        x = self.output(x)
        
        return x


class LightAttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 3) -> None:
        """
        Light Attention U-Net architecture for semantic segmentation.

        Args:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels (classes).

        Returns:
            - None.
        """
        super(LightAttentionUNet, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_1 = ConvBlock(in_channels=in_channels, output_channels=64, kernel_size=3, padding=1)
        self.down_2 = ConvBlock(in_channels=64, output_channels=128, kernel_size=3, padding=1)
        self.down_3 = ConvBlock(in_channels=128, output_channels=256, kernel_size=3, padding=1)
        self.down_4 = ConvBlock(in_channels=256, output_channels=512, kernel_size=3, padding=1)
        
        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.attention_block_1 = AttentionGate(g_in_channels=512, x_in_channels=256)
        self.up_conv_1 = ConvBlock(in_channels=512, output_channels=256, kernel_size=3, padding=1)

        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.attention_block_2 = AttentionGate(g_in_channels=256, x_in_channels=128)
        self.up_conv_2 = ConvBlock(in_channels=256, output_channels=128, kernel_size=3, padding=1)

        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.attention_block_3 = AttentionGate(g_in_channels=128, x_in_channels=64)
        self.up_conv_3 = ConvBlock(in_channels=128, output_channels=64, kernel_size=3, padding=1)

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
        g = self.down_4(x6)
          
        out = self.up_1(g)
        x5_caret = self.attention_block_1(g, x5)
        g = self.up_conv_1(torch.cat([x5_caret, out], dim=1))                
        out = self.up_2(g)
        x3_caret = self.attention_block_2(g, x3)        
        g = self.up_conv_2(torch.cat([x3_caret, out], dim=1))       
        out = self.up_3(g)
        x1_caret = self.attention_block_3(g, x1)
        x = self.up_conv_3(torch.cat([x1_caret, out], dim=1))

        x = self.output(x)
        
        return x


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
        self.attention_block_1 = AttentionGate(g_in_channels=1024, x_in_channels=512)
        self.up_conv_1 = ConvBlock(in_channels=1024, output_channels=512, kernel_size=3, padding=1)

        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.attention_block_2 = AttentionGate(g_in_channels=512, x_in_channels=256)
        self.up_conv_2 = ConvBlock(in_channels=512, output_channels=256, kernel_size=3, padding=1)

        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.attention_block_3 = AttentionGate(g_in_channels=256, x_in_channels=128)
        self.up_conv_3 = ConvBlock(in_channels=256, output_channels=128, kernel_size=3, padding=1)

        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.attention_block_4 = AttentionGate(g_in_channels=128, x_in_channels=64)
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
        g = self.down_5(x8)
       
        out = self.up_1(g)
        x7_caret = self.attention_block_1(g=g, x=x7)
        g = self.up_conv_1(torch.cat([x7_caret, out], dim=1))      
        out = self.up_2(g)   
        x5_caret = self.attention_block_2(g, x5)
        g = self.up_conv_2(torch.cat([x5_caret, out], dim=1))                
        out = self.up_3(g)
        x3_caret = self.attention_block_3(g, x3)        
        g = self.up_conv_3(torch.cat([x3_caret, out], dim=1))       
        out = self.up_4(g)
        x1_caret = self.attention_block_4(g, x1)
        x = self.up_conv_4(torch.cat([x1_caret, out], dim=1))

        x = self.output(x)
        
        return x


""" def prepare_model(num_classes=3):
    model = deeplabv3_resnet101(weights="DEFAULT")
    
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    
    return model
"""
