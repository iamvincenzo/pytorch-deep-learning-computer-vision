import torchvision
import torchvision.models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights


def create_model(num_classes):
    """
    Creates a Faster R-CNN model for finetuning with a custom number of classes.

    Parameters:
        - num_classes (int): Number of output classes for the detector.

    Returns:
        - model (torchvision.models.detection.FasterRCNN): Faster R-CNN model with a modified
          box predictor head for the specified number of classes.
    """    
    # load Faster RCNN pre-trained model with MobileNetV3 backbone
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(trainable_backbone_layers=3,
    #                                                                        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    # # # alternatively, you can use the ResNet50 backbone:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # define a new head for the detector with the required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, 
                                                      num_classes=num_classes)

    return model
