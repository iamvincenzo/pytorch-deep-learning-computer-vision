"""
Convert labelme annotations to masks (h,w,1), where each element of the matrix is the id of the class
class 0: backgorund
class 1: foliage
class 2: waste
"""

# MODIFIED
#################################################################################################
import os
import cv2
import json
import numpy as np
from mask_utils import Mask
import matplotlib.pyplot as plt

IMG_RES_W = 608
IMG_RES_H = 416

SAVE_PTH = "./data/images/"
SRC_PATH = "./data/front/"

if __name__ == "__main__":
    m = Mask()
    
    if not os.path.exists(SAVE_PTH):
        os.mkdir(SAVE_PTH)

    for filename in os.listdir(SRC_PATH):
        if filename.endswith(".json"):
            img_name = filename.split('.')[0] + '.jpg'
            img = cv2.imread(SRC_PATH + img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv open images in BGR
            img_w = img.shape[1]
            img_h = img.shape[0]

            # get polygons from json file
            with open(SRC_PATH + filename, 'r') as f:
                dataset = json.loads(f.read())
                annots = dataset["shapes"]
                class1_annots = [x for x in annots if x['label'] == "foliage"]
                class2_annots = [x for x in annots if x['label'] == "waste"]

            polygons1 = []
            for class1_annot in class1_annots:
                s = class1_annot["points"]
                polygon = m.get_xy_segmentation(s)
                polygons1.append(polygon)

            polygons2 = []
            for class2_annot in class2_annots:
                s = class2_annot["points"]
                polygon = m.get_xy_segmentation(s)
                polygons2.append(polygon)

            mask = m.create_masks_multiclass(polygons1, polygons2, img.shape[0], img.shape[1])

            img_res = cv2.resize(img, (IMG_RES_W, IMG_RES_H)) # RGB format
            mask_res = cv2.resize(mask, (IMG_RES_W, IMG_RES_H)) # BGR format
            mask_res = m.create_masks_one_hot(mask_res) # one-hot mask
            mask_res = np.argmax(mask_res, axis=-1) # one-hot mask

            # back to BGR before save
            img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(SAVE_PTH + filename.split('.')[0] + '.png', img_res)
            cv2.imwrite(SAVE_PTH + filename.split('.')[0] + '_mask.png', mask_res)
            
            # plt.subplot(1, 2, 1);  plt.imshow(img_res)
            # plt.subplot(1, 2, 2);  plt.imshow(mask_res)
            # plt.show()
#################################################################################################
            