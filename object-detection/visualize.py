import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from PIL import ImageDraw
import matplotlib.pyplot as plt
from torchvision.transforms import transforms


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

    # read and filter testing metadata (remove images that don't exist)
    test_df = pd.read_csv(filepath_or_buffer=METADATA_TEST_PATH)
    filtered_test_df = test_df[[os.path.isfile(TEST_IMGS_PATH / filename) for filename in test_df["filename"]]]

    # print(min(filter(lambda x: x > 0, filtered_train_df["width"])))
    # print(min(filter(lambda x: x > 0, filtered_train_df["height"])))
    # print(max(filtered_train_df["width"]))
    # print(max(filtered_train_df["height"]))

    filename, w, h, _, xmin, ymin, xmax, ymax = train_df.iloc[0]

    w_ratio = RESIZE / w
    h_ratio = RESIZE / h

    img = Image.open(fp=TRAIN_IMGS_PATH/filename)

    draw = ImageDraw.Draw(img)

    rects = np.array(filtered_train_df[filtered_train_df["filename"] == filename].iloc[:, 4:])

    for xmin, ymin, xmax, ymax in rects:
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=1)

    plt.title("Original size")
    plt.imshow(img)
    plt.show()

    transform = transforms.Compose([transforms.Resize((RESIZE, RESIZE))])
    img = transform(img)

    for xmin, ymin, xmax, ymax in rects:
        xmin = int(xmin * w_ratio)
        ymin = int(ymin * h_ratio)
        xmax = int(xmax * w_ratio)
        ymax = int(ymax * h_ratio)
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=1)

    plt.title("Resized")
    plt.imshow(img)
    plt.show()