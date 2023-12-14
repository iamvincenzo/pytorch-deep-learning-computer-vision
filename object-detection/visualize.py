import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from main import clean_dataframe
from data_augmentation import CustomAlbumentations

# parameters
RESIZE = 224

TEST_IMGS_PATH = Path("./data/TEST")
TRAIN_IMGS_PATH = Path("./data/TRAIN")
METADATA_TEST_PATH = Path("./data/test_labels.csv")
METADATA_TRAIN_PATH = Path("./data/train_labels.csv")


if __name__ == "__main__":
    train_file_path = "./data/train_labels_fixed.csv"
    if not os.path.exists(train_file_path):
        # read train metadata
        train_df = pd.read_csv(filepath_or_buffer=METADATA_TRAIN_PATH)
        # clean train dataframe
        train_df = clean_dataframe(df=train_df, imgs_root_path=TRAIN_IMGS_PATH, save_path=train_file_path)
    else:
        # read train metadata
        train_df = pd.read_csv(filepath_or_buffer=train_file_path)
    
    all_images = train_df["filename"].unique().tolist()

    albument = CustomAlbumentations(resize_h=RESIZE,
                                    resize_w=RESIZE,
                                    transform="basic")

    for filename in all_images:
        img = cv2.imread(str(TRAIN_IMGS_PATH / filename)) # OK
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OK
        bboxes = torch.from_numpy(np.array(train_df[train_df["filename"] == filename].iloc[:, 4:])).float()
        class_labels = [name for name in train_df[train_df["filename"] == filename].iloc[:, 3]]
        height, width, _ = img.shape

        print(f"\n{filename}\n")
        print(f"{bboxes}")
        print(f"{class_labels}")
        print(f"{height, width}")

        # normalize bounding box coordinates
        normalized_bboxes = bboxes.clone()
        normalized_bboxes[:, 0] /= width # xmin
        normalized_bboxes[:, 1] /= height # ymin
        normalized_bboxes[:, 2] /= width # xmax
        normalized_bboxes[:, 3] /= height # ymax

        print(f"normalized_bboxes: {normalized_bboxes}")

        # apply augmentations
        # Albumentation ToTensorV2 do not perform normalization automatically 
        img = img.astype(dtype=np.float32) / 255.0
        transformed_img, transformed_bboxes, class_labels = albument(image=img,
                                                                     bboxes=normalized_bboxes,
                                                                     class_labels=class_labels)
        transformed_img = (transformed_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(dtype=np.uint8)

        width, height = RESIZE, RESIZE
        # Convert transformed bounding boxes back to pixel coordinates for visualization
        transformed_bboxes[:, 0] *= width # xmin
        transformed_bboxes[:, 1] *= height # ymin
        transformed_bboxes[:, 2] *= width # xmax
        transformed_bboxes[:, 3] *= height # ymax

        for bbox in transformed_bboxes:
            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(transformed_img, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1)

        plt.title("Augmented image")
        plt.imshow(transformed_img)
        plt.show()
