import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


# reproducibility
SEED = 42
random.seed(SEED)


class CustomAlbumentations(object):
    def __init__(self, resize_h, resize_w, transform="basic"):
        """
        CustomAlbumentations is a class that encapsulates image augmentation operations using Albumentations library.

        Args:
            - resize (int): The target size for resizing images.
            - transform (albumentations.Compose, optional): Custom transformation to be applied. 
                If None, a default set of augmentations is used.
        """
        super(CustomAlbumentations, self).__init__()
        self.resize_h = resize_h
        self.resize_w = resize_w

        # set the transformation pipeline
        if transform == "basic":
            self.transform = A.Compose([
                A.Resize(height=self.resize_h, width=self.resize_w),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="albumentations",
                                        min_area=0,
                                        min_visibility=0,
                                        label_fields=["class_labels"]))

        elif transform == "advance":
            self.transform = A.Compose([
                # A.RandomCrop(width=self.resize_h, height=self.resize_w),
                A.Resize(height=self.resize_h, width=self.resize_w),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # A.Rotate(limit=(-30, 30), border_mode=cv2.BORDER_CONSTANT, value=.0, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
                A.Blur(blur_limit=(3, 7), p=0.5),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="albumentations",
                                        min_area=0, 
                                        min_visibility=0,
                                        label_fields=["class_labels"]))

    def __call__(self, image, masks, bboxes, class_labels):    
        """
        Apply the defined transformations to the input image, bounding boxes, and class labels.

        Args:
            - image (numpy.ndarray): Input image.
            - bboxes (list of tuples): Bounding boxes in the format [(x_min, y_min, x_max, y_max), ...].
            - class_labels (list): List of class labels corresponding to each bounding box.

        Returns:
            - transformed_image (numpy.ndarray): Transformed image.
            - transformed_bboxes (list of tuples): Transformed bounding boxes.
            - transformed_class_labels (list): Transformed class labels.
        """
        transformed = self.transform(image=image,
                                     masks=masks,
                                     bboxes=bboxes, 
                                     class_labels=class_labels)
        transformed_image = transformed["image"]
        transformed_masks = transformed["masks"]
        transformed_bboxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        transformed_class_labels = torch.as_tensor(transformed["class_labels"], dtype=torch.int64)

        return transformed_image, transformed_masks, transformed_bboxes, transformed_class_labels
