import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights


def create_model(num_classes):
    """
    Creates a Mask R-CNN model for finetuning with a custom number of classes.

    Parameters:
        - num_classes (int): Number of output classes for the detector.

    Returns:
        - model (torchvision.models.detection.FasterRCNN): Faster R-CNN model with a modified
          box predictor head for the specified number of classes.
    """
    # load MaskRCNN pre-trained model with ResNet50 backbone
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes=num_classes)

    return model
