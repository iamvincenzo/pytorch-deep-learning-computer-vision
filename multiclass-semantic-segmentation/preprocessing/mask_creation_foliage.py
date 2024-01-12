"""
Binary segmentation - only foliage
Convert labelme annotations to masks (h,w,1), where each element of the matrix is the id of the class
class 0: backgorund
class 1: foliage
"""

import json
import numpy as np
import cv2
from mask_utils import Mask
import os
import matplotlib.pyplot as plt

m = Mask()

IMG_RES_W = 608
IMG_RES_H = 416

dst_dir_arr = "saved_arrays_foliage"
if not os.path.exists(dst_dir_arr):
    os.mkdir(dst_dir_arr)

def process_image_and_mask(src_path):
    print(src_path)
    images = []
    masks = []
    for filename in os.listdir(src_path):
        if filename.endswith(".json"):
            img_name = filename.split('.')[0] + '.jpg'
            img = cv2.imread(src_path + img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv open images in BGR

            # get polygons from json file
            with open(src_path + filename, 'r') as f:
                dataset = json.loads(f.read())
                annots = dataset["shapes"]
                class1_annots = [x for x in annots if x['label'] == "foliage"]

            if len(class1_annots)>0:
                polygons1 = []
                for class1_annot in class1_annots:
                    s = class1_annot["points"]
                    polygon = m.get_xy_segmentation(s)
                    polygons1.append(polygon)

                mask = m.create_masks_binary(polygons1, img.shape[0], img.shape[1])
                #plt.imshow(mask)
                #plt.show()
                img_res = cv2.resize(img, (IMG_RES_W, IMG_RES_H))
                images.append(img_res)
                mask_res = cv2.resize(mask, (IMG_RES_W, IMG_RES_H))
                masks.append(mask_res)

    images = np.array(images)
    masks = np.array(masks)

    return images, masks


images1, masks1 = process_image_and_mask("../frames_BT_upload/1/front_SN214929_2023_10_17_0733_0001/")
images2, masks2 = process_image_and_mask("../frames_BT_upload/2/front_SN214929_2023_10_17_0743_0001/")
images3, masks3 = process_image_and_mask("../frames_BT_upload/3/front_SN214929_2023_10_17_1027_0001/")
images4, masks4 = process_image_and_mask("../frames_BT_upload/4/front_SN214929_2023_10_17_1147_0001/")
images5, masks5 = process_image_and_mask("../frames_BT_upload/5/front_SN214929_2023_10_17_1157_0001/")
images6, masks6 = process_image_and_mask("../frames_BT_upload/6/left_nozzle_SN214929_2023_11_06_0714_0001/")
images7, masks7 = process_image_and_mask("../frames_BT_upload/7/left_nozzle_SN214929_2023_11_03_1005_0001/")
images8, masks8 = process_image_and_mask("../frames_BT_upload/8/left_nozzle_SN214929_2023_11_03_1316_0001/")
images9, masks9 = process_image_and_mask("../frames_BT_upload/9/left_nozzle_SN214929_2023_11_03_0955_0001/")

images_failure, masks_failure = process_image_and_mask("../frames/front_failure/")
images0_in, masks0_in = process_image_and_mask("../frames/front0/")
images2_in, masks2_in = process_image_and_mask("../frames/front2/")

images_train = np.concatenate((images1,images2,images3,images4,images5,images6,images7,images8,images9,images_failure), axis=0)
masks_train = np.concatenate((masks1,masks2,masks3,masks4,masks5,masks6,masks7,masks8,masks9,masks_failure), axis=0)

images_test = np.concatenate((images0_in,images2_in), axis=0)
masks_test = np.concatenate((masks0_in,masks2_in), axis=0)

print("images_train", images_train.shape)
print("masks_train", masks_train.shape)
print("images_test", images_test.shape)
print("masks_test", masks_test.shape)

np.save(os.path.join(dst_dir_arr, "images_train.npy"), images_train)
np.save(os.path.join(dst_dir_arr, "masks_train.npy"), masks_train)
np.save(os.path.join(dst_dir_arr, "images_test.npy"), images_test)
np.save(os.path.join(dst_dir_arr, "masks_test.npy"), masks_test)
