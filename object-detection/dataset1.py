"""
PlantDocDataset: A PyTorch Dataset for the PlantDoc Object Detection dataset.

Dataset Source: https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset
Research Paper: https://arxiv.org/pdf/1911.10317.pdf

This dataset is designed for visual plant disease detection, containing images
of plants with associated bounding box annotations for various plant diseases.
"""
import os
import cv2
import torch
import random
import numpy as np
import glob as glob
from torch.utils.data import Dataset
from data_augmentation import CustomAlbumentations
from xml.etree import ElementTree as et # to load bbox files


# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


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
    def __init__(self, dir_path, resize_h, resize_w, classes, data_aug):
        """
        Initializes the PlantDocDataset.

        Parameters:
            - dir_path (str): The path to the directory containing the dataset.
            - resize_h (int): The height to which the images will be resized.
            - resize_w (int): The width to which the images will be resized.
            - classes (list): A list of classes present in the dataset.
            - data_aug (bool): A flag indicating whether data augmentation should be applied.

        Returns:
            - None
        """
        super(PlantDocDataset, self).__init__()
        self.classes = classes
        self.dir_path = dir_path
        self.resize_h = resize_h
        self.resize_w = resize_w

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/JPEGImages/*.jpg")
        # extract image name removing all absolute/relative path
        self.all_images = sorted([image_path.split(os.path.sep)[-1] for image_path in self.image_paths])
        
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
        # capture the image name and the full image path
        image_name = self.all_images[index]
        image_path = os.path.join(self.dir_path, "JPEGImages", image_name)
        # read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # capture the corresponding `.xml` file for getting the annotations
        annot_filename = image_name[:-4] + ".xml"
        annot_file_path = os.path.join(self.dir_path, "Annotations", annot_filename)

        bboxes, labels = [], []

        # parsing the XML file specified by annot_file_path
        tree = et.parse(annot_file_path)
        # getting the root element of the XML tree
        root = tree.getroot()

        # get the height and width of the image
        image_height, image_width, _ = image.shape

        # for each object starting from the root of the `.xml` file
        for member in root.findall("object"):
            # get the index of the corresponding class in `classes` list to get the label
            labels.append(self.classes.index(member.find("name").text))
            
            # xmin: left corner x-coordinates
            xmin = int(member.find("bndbox").find("xmin").text)
            # xmax: right corner x-coordinates
            xmax = int(member.find("bndbox").find("xmax").text)
            # ymin: left corner y-coordinates
            ymin = int(member.find("bndbox").find("ymin").text)
            # ymax: right corner y-coordinates
            ymax = int(member.find("bndbox").find("ymax").text)
            
            # normalize bounding box coordinates
            xmin_final = xmin / image_width
            xmax_final = xmax / image_width
            ymin_final = ymin / image_height
            ymax_final = ymax / image_height
            
            bboxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # bounding box to tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        # area of the bounding boxes (ymax-ymin)*(xmax -xmin) = height*width
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # apply augmentations
        # Albumentation ToTensorV2 do not perform normalization automatically 
        image = image.astype(dtype=np.float32) / 255.0
        image, bboxes, labels = self.transform(image=image,
                                               bboxes=bboxes,
                                               class_labels=labels)

        # prepare the final `target` dictionary
        target = {}
        target["area"] = area
        target["boxes"] = bboxes
        target["labels"] = labels
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([index])

        # # DEBUG
        # # convert transformed bounding boxes back to pixel coordinates for visualization (denormalize)
        # import matplotlib.pyplot as plt
        # np_img = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(dtype=np.uint8)
        # bboxes[:, 0] *= self.resize_w  # xmin
        # bboxes[:, 2] *= self.resize_w  # xmax
        # bboxes[:, 1] *= self.resize_h  # ymin
        # bboxes[:, 3] *= self.resize_h  # ymax
        # for bbox, label in zip(bboxes, labels):
        #     xmin, ymin, xmax, ymax = map(int, bbox)
        #     cv2.rectangle(np_img, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1)
        #     # put class label on the top left of the rectangle
        #     label_text = f"Class: {self.classes[label]}"
        #     cv2.putText(np_img, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # plt.imshow(np_img)
        # plt.show()

        return image, target

    def __len__(self):
        """
        Returns the total number of unique images in the dataset.

        Returns:
            - int: The number of unique images in the dataset.
        """
        return len(self.all_images)


# # TEST: PLANTDOC
# import pandas as pd
# from torch.utils.data import DataLoader
# RESIZE = 224
# BATCH_SIZE = 1
# TEST_IMGS_PATH = "./data/TEST"
# TRAIN_IMGS_PATH = "./data/TRAIN"
# METADATA_TRAIN_PATH = "./data/train_labels.csv"
# if __name__ == "__main__":
#     train_df = pd.read_csv(filepath_or_buffer=METADATA_TRAIN_PATH)
#     classes = ["__background__"] + sorted(train_df["class"].unique().tolist())

#     train_set = PlantDocDataset(dir_path=TRAIN_IMGS_PATH,
#                                 resize_h=RESIZE, resize_w=RESIZE, classes=classes, data_aug=True)
#     test_set = PlantDocDataset(dir_path=TEST_IMGS_PATH,
#                                resize_h=RESIZE, resize_w=RESIZE, classes=classes, data_aug=False)

#     train_loader = DataLoader(dataset=train_set, 
#                               batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
#     test_loader = DataLoader(dataset=test_set, 
#                              batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#     for i, data in enumerate(train_loader):
#         print(f"Batch id: {i}")
        
#     for i, data in enumerate(train_loader):
#         print(f"Batch id: {i}")


# TEST: PERSONS
from torch.utils.data import DataLoader
from solver import Solver
from models import create_model
# parameters
EPOCHS = 200
RESIZE = 224
PATIENCE = 5
BATCH_SIZE = 16
TEST_IMGS_PATH = "./data/persons/Test/Test"
TRAIN_PATH = "./data/persons/Train/Train"
RESUME_TRAIN = True
if __name__ == "__main__":
    classes = ["__background__", "person", "person-like"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = PlantDocDataset(dir_path=TRAIN_PATH,
                                resize_h=RESIZE, resize_w=RESIZE, classes=classes, data_aug=False)
    test_set = PlantDocDataset(dir_path=TEST_IMGS_PATH,
                               resize_h=RESIZE, resize_w=RESIZE, classes=classes, data_aug=False)
    
    # determine if GPU pinning is possible
    pin = True if torch.cuda.is_available() else False

    train_loader = DataLoader(dataset=train_set, pin_memory=pin,
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_set, pin_memory=pin,
                             batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
 
    # initialize the model and move to the computation device
    model = create_model(num_classes=len(classes))

    if RESUME_TRAIN:
        model.load_state_dict(torch.load("./checkpoints/model-epoch=3-val_loss=0.1809.pth", 
                                         map_location=device))

    model = model.to(device)

    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # create the Solver instance
    solver = Solver(epochs=EPOCHS, 
                    device=device, 
                    train_loader=train_loader, 
                    valid_loader=test_loader, 
                    model=model, 
                    optimizer=optimizer,
                    patience=PATIENCE)
    
    # train the neural network
    solver.train_net()
