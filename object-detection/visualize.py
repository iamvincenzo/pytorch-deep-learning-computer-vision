import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from PIL import ImageDraw
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from data_augmentation import resize_bbox
from data_augmentation import custom_transform


# # reproducibility
# SEED = 42
# random.seed(SEED)

# parameters
RESIZE = 128

TEST_IMGS_PATH = Path("./data/TEST")
TRAIN_IMGS_PATH = Path("./data/TRAIN")
METADATA_TEST_PATH = Path("./data/test_labels.csv")
METADATA_TRAIN_PATH = Path("./data/train_labels.csv")


if __name__ == "__main__":
    # read and filter training metadata (remove images that don't exist)
    train_df = pd.read_csv(filepath_or_buffer=METADATA_TRAIN_PATH)
    filtered_train_df = train_df[[os.path.isfile(TRAIN_IMGS_PATH / filename) for filename in train_df["filename"]]]

    filename, w, h, *_ = train_df.iloc[0]
    bboxes = torch.from_numpy(np.array(filtered_train_df[filtered_train_df["filename"] == filename].iloc[:, 4:]))

    img = Image.open(fp=TRAIN_IMGS_PATH/filename)
    
    w_ratio = RESIZE / w
    h_ratio = RESIZE / h
    # resize flipped bounding boxes
    bboxes = resize_bbox(bboxes=bboxes, w_ratio=w_ratio, h_ratio=h_ratio)
    img = TF.resize(img=img, size=(RESIZE, RESIZE))

    for _ in range(100):
        img_aug, bboxes_aug = custom_transform(img=img, bboxes=bboxes)
        draw = ImageDraw.Draw(img_aug)

        for xmin, ymin, xmax, ymax in bboxes_aug:
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=1)

        plt.title("Augmented image")
        plt.imshow(img_aug)
        plt.show()
