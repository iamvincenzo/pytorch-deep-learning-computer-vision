import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights


# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# parameters
BS = 1
RESIZE = 224
NUM_CLASSES = 1
LABELS = {0: "Clean", 1: "Dirty"}
TEST_SET_PTH = Path("./data/test")
MODEL_PATH = Path("./checkpoints/model.pth")


def get_prediction(model, x_test):
    """
    Get model prediction for a given input.

    Args:
        model (torch.nn.Module): The trained model.
        x_test (torch.Tensor): Input tensor for prediction.

    Returns:
        Tuple[int, float]: Predicted label and probability.
    """
    model.eval()
    with torch.inference_mode():
        x_test = x_test.unsqueeze(0).to(device)
        logits = model(x_test)
        prob = torch.sigmoid(logits)
        y_pred = torch.round(prob)

    return y_pred.cpu().item(), prob.cpu().item()


def plot_prediction(img, pred_label, prob, true_label):
    """
    Plot the model prediction along with true label on the input image.

    Args:
        img (PIL.Image.Image): Input image.
        pred_label (int): Predicted label.
        prob (float): Prediction probability.
        true_label (int): True label.
    """
    if pred_label == true_label:
        color = (0, 255, 0)  # green for correct predictions
    else:
        color = (255, 0, 0)  # red for incorrect predictions

    im_draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 18)

    # display prediction information on the image
    prob = ((1 - prob) * 100) if pred_label == 0 else (prob * 100)
    im_draw.text((15, 15), f"Predicted label: {LABELS[pred_label]}, {prob:.2f}%", font=font, fill=color)
    im_draw.text((30, 30), f"True label: {LABELS[true_label]}", font=font, fill=(255, 255, 255))

    # display the image with prediction information
    plt.title(f"Pred: {LABELS[pred_label]}, {prob:.2f}% - True: {LABELS[true_label]}")
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pre-trained ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    nr_filters = model.fc.in_features
    model.fc = nn.Linear(nr_filters, NUM_CLASSES)
    model.load_state_dict(torch.load(f=MODEL_PATH, map_location=device))

    # define the transformation for inference
    inference_transform = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # load the test dataset
    test_set = ImageFolder(root=TEST_SET_PTH)

    # perform inference on each image in the test set
    for _, (img, label) in enumerate(test_set):
        tensor = inference_transform(img)
        y_pred, prob = get_prediction(model=model, x_test=tensor)
        plot_prediction(img=img, pred_label=y_pred, prob=prob, true_label=label)
