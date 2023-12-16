import os
import cv2
import torch
import random
import numpy as np
import glob as glob
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
# from data_augmentation import CustomAlbumentations


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


class PennFudanPedDataset(Dataset):
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
        super(PennFudanPedDataset, self).__init__()
        self.classes = classes
        self.dir_path = dir_path
        self.resize_h = resize_h
        self.resize_w = resize_w

        # extract image name removing all absolute/relative path
        self.all_images = sorted([img for img in os.listdir(os.path.join(self.dir_path, "PNGImages"))])
        
        # transform = "advance" if data_aug else "basic"
        # self.transform = CustomAlbumentations(resize_h=self.resize_h, 
        #                                       resize_w=self.resize_w, 
        #                                       transform=transform)
    
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
        image_path = os.path.join(self.dir_path, "PNGImages", image_name)
        # read the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # capture the corresponding mask
        mask_name = image_name[:-4] + "_mask" + image_name[-4:]
        mask_path = os.path.join(self.dir_path, "PedMasks", mask_name)
        # read the mask image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # get object ids excluding background id (0) 
        obj_ids = np.unique(mask)[1:]
        num_objs = len(obj_ids)

        # create a segmentation-mask for each object 
        masks = np.zeros((num_objs , mask.shape[0] , mask.shape[1]))
        for i in range(num_objs):
            masks[i][mask == i+1] = True   

        # get the height and width of the image
        image_height, image_width, _ = image.shape

        # get bounding boxes using extreme coordinates (top, bottom, left, right)
        bboxes, labels = [], []

        for i in range(num_objs):
            # get the index of the corresponding class in `classes` list to get the label
            labels.append(self.classes.index("PASpersonWalking"))

            # returns tuple with x and y coordinates
            pos = np.where(masks[i])
            # normalize bounding box coordinates
            xmin = np.min(pos[1]) / image_width
            xmax = np.max(pos[1]) / image_width
            ymin = np.min(pos[0]) / image_height
            ymax = np.max(pos[0]) / image_height

            bboxes.append([xmin, ymin, xmax, ymax])

        # bounding box to tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # image to tensor
        image = T.ToTensor()(image)
        # masks to tensor
        masks = torch.as_tensor(masks , dtype = torch.uint8)

        # # apply augmentations
        # # Albumentation ToTensorV2 do not perform normalization automatically
        # https://www.kaggle.com/code/blondinka/how-to-do-augmentations-for-instance-segmentation
        # image = image.astype(dtype=np.float32) / 255.0
        # masks = masks.astype(dtype=np.uint8)
        # image, masks, bboxes, labels = self.transform(image=image,
        #                                               masks=masks,
        #                                               bboxes=bboxes,
        #                                               class_labels=labels)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = masks

        # # DEBUG
        # # convert transformed bounding boxes back to pixel coordinates for visualization (denormalize)
        # import matplotlib.pyplot as plt
        # np_img = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(dtype=np.uint8).copy()
        # self.resize_h, self.resize_w = image_height, image_width
        # bboxes[:, 0] *= self.resize_w # xmin
        # bboxes[:, 2] *= self.resize_w # xmax
        # bboxes[:, 1] *= self.resize_h # ymin
        # bboxes[:, 3] *= self.resize_h # ymax
        # for bbox, label in zip(bboxes, labels):
        #     xmin, ymin, xmax, ymax = map(int, bbox)
        #     cv2.rectangle(img=np_img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(255, 0, 0), thickness=1)
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
