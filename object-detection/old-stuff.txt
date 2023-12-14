"""
PlantDocDataset: A PyTorch Dataset for the PlantDoc Object Detection dataset.

Dataset Source: https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset
Research Paper: https://arxiv.org/pdf/1911.10317.pdf

This dataset is designed for visual plant disease detection, containing images
of plants with associated bounding box annotations for various plant diseases.
"""

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


# parameters
SKIPCOLS = 4  # number of columns to skip in the DataFrame for extracting bounding box coordinates
CLASS_COL = 3  # column index containing class labels in the DataFrame


def collate_fn(batch):
    """
    Custom collate function to handle data loading with varying numbers of objects
    and varying size tensors.

    Parameters:
        - batch (list): A list of tuples containing image tensors and annotation targets.

    Returns:
        - tuple: A tuple containing two lists - one for images and one for annotation targets.
    """
    return tuple(zip(*batch))


class PlantDocDataset(Dataset):
    def __init__(self, df, transform, resize, root, classes, classes_inverse):
        """
        Initializes the PlantDocDataset.

        Parameters:
            - df (pandas.DataFrame): The DataFrame containing metadata about images and annotations.
            - transform (torchvision.transforms.Compose): A composition of image transformations.
            - resize (int): The target size for resizing the images.
            - root (Path): The root path to the dataset images.
            - classes (dict): A dictionary mapping class names to numerical labels.
            - classes_inverse (dict): A dictionary mapping numerical labels to class names.
        """
        super(PlantDocDataset, self).__init__()
        self.df = df
        self.root = root  # root path to the dataset images
        self.resize = resize  # target size for resizing the images
        self.classes = classes  # dictionary mapping class names to numerical labels
        self.classes_inverse = classes_inverse  # dictionary mapping numerical labels to class names
        self.all_images = self.df["filename"].unique().tolist()

        # check if a custom transformation is provided, otherwise use default
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        """
        Retrieves and processes a single image and its associated annotations.

        Parameters:
            - index (int): Index of the image in the dataset.

        Returns:
            - tuple: A tuple containing the processed image tensor and annotation target dictionary.
        """
        filename = self.all_images[index]  # get the filename of the current image
        img = Image.open(fp=self.root / filename).convert("RGB")  # open the image using PIL
        width, height = img.size  # get the original width and height of the image

        img_t = self.transform(img)  # apply the specified image transformations

        # extract bounding box coordinates and class labels from the DataFrame
        boxes = np.array(self.df[self.df["filename"] == filename].iloc[:, SKIPCOLS:])
        classes = np.array([self.classes[name] for name in
                            self.df[self.df["filename"] == filename].iloc[:, CLASS_COL]])
        boxes_t = torch.from_numpy(boxes).to(dtype=torch.float32)  # convert bounding boxes to PyTorch tensor
        classes_t = torch.from_numpy(classes).to(dtype=torch.int64)  # convert class labels to PyTorch tensor

        w_ratio = self.resize / width  # calculate width ratio for normalization
        h_ratio = self.resize / height  # calculate height ratio for normalization

        # normalize bounding box coordinates based on image size
        boxes_t[:, 0] = boxes_t[:, 0] * w_ratio
        boxes_t[:, 2] = boxes_t[:, 2] * w_ratio
        boxes_t[:, 1] = boxes_t[:, 1] * h_ratio
        boxes_t[:, 3] = boxes_t[:, 3] * h_ratio

        target = {}
        target["boxes"] = boxes_t
        target["labels"] = classes_t

        return img_t, target

    def __len__(self):
        """
        Returns the total number of unique images in the dataset.

        Returns:
            - int: The number of unique images in the dataset.
        """
        return len(self.df["filename"].unique())
