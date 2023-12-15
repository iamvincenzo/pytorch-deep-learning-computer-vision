"""
PlantDocDataset: A PyTorch Dataset for the PlantDoc Object Detection dataset.

Dataset Source: https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset
Research Paper: https://arxiv.org/pdf/1911.10317.pdf

This dataset is designed for visual plant disease detection, containing images
of plants with associated bounding box annotations for various plant diseases.
"""
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from data_augmentation import CustomAlbumentations


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
    def __init__(self, df, data_aug, resize_h, resize_w, root, classes, classes_inverse):
        """
        Initializes the PlantDocDataset.

        Parameters:
            - df (pandas.DataFrame): The DataFrame containing metadata about images and annotations.
            # - data_aug (Bool): A composition of image transformations.
            - resize (int): The target size for resizing the images.
            - root (Path): The root path to the dataset images.
            - classes (dict): A dictionary mapping class names to numerical labels.
            - classes_inverse (dict): A dictionary mapping numerical labels to class names.
        """
        super(PlantDocDataset, self).__init__()
        self.df = df
        self.root = root # root path to the dataset images
        self.resize_h = resize_h # target size for resizing the images
        self.resize_w = resize_w # target size for resizing the images
        self.classes = classes # dictionary mapping class names to numerical labels
        self.classes_inverse = classes_inverse # dictionary mapping numerical labels to class names
        self.all_images = self.df["filename"].unique().tolist()

        transform = "advance" if data_aug else "basic"
        self.transform = CustomAlbumentations(resize_h=self.resize_h, 
                                              resize_w=self.resize_w, 
                                              transform=transform)

    def __getitem__(self, index):
        """
        Retrieves and processes a single image and its associated annotations.

        Parameters:
            - index (int): Index of the image in the dataset.

        Returns:
            - tuple: A tuple containing the processed image tensor and annotation target dictionary.
        """
        filename = self.all_images[index]  # get the filename of the current image
        img = cv2.imread(str(self.root/filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape  # get the original width and height of the image

        # extract bounding box coordinates and class labels from the DataFrame
        bboxes = torch.from_numpy(np.array(self.df[self.df["filename"] == filename].iloc[:, SKIPCOLS:])).to(dtype=torch.float32)
        classes = torch.from_numpy(np.array([self.classes[name] for name in
                                             self.df[self.df["filename"] == filename].iloc[:, CLASS_COL]])).to(dtype=torch.int64)
        class_labels = [name for name in self.df[self.df["filename"] == filename].iloc[:, CLASS_COL]]

        # normalize bounding box coordinates
        bboxes[:, 0] /= width # xmin
        bboxes[:, 1] /= height # ymin
        bboxes[:, 2] /= width # xmax
        bboxes[:, 3] /= height # ymax

        # apply augmentations
        # Albumentation ToTensorV2 do not perform normalization automatically 
        img = img.astype(dtype=np.float32) / 255.0
        img, bboxes, class_labels = self.transform(image=img,
                                                   bboxes=bboxes,
                                                   class_labels=class_labels)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = classes

        # # DEBUG
        # # convert transformed bounding boxes back to pixel coordinates for visualization
        # np_img = img.numpy().transpose(1, 2, 0)
        # height, width = self.resize_h, self.resize_w
        # bboxes[:, 0] *= width # xmin
        # bboxes[:, 1] *= height # ymin
        # bboxes[:, 2] *= width # xmax
        # bboxes[:, 3] *= height # ymax
        # for bbox in bboxes:
        #     xmin, ymin, xmax, ymax = map(int, bbox)
        #     cv2.rectangle(np_img, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1)
        # plt.title("Augmented image")
        # plt.imshow(np_img)
        # plt.show()

        return img, target

    def __len__(self):
        """
        Returns the total number of unique images in the dataset.

        Returns:
            - int: The number of unique images in the dataset.
        """
        return len(self.df["filename"].unique())
