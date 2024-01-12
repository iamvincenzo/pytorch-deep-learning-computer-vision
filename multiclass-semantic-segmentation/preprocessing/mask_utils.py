from PIL import Image, ImageDraw
import numpy as np


class Mask:
    def __init__(self):
        self.color1 = [0, 255, 0]  # color class 1
        self.color2 = [255, 0, 0]  # color class 2


    def get_xy_segmentation(self, segment):
        """
        Transform polygon into [x1, y1, x2, y2, ..., xn, yn] format
        :param segment: segmentation in labelme format
        :return: polygon [x1, y1, x2, y2, ..., xn, yn]
        """
        polygon = []
        for p in segment:
            polygon.append(p[0])
            polygon.append(p[1])
        return polygon


    def create_masks_binary(self, polygons, img_h, img_w):
        """
        create mask with more polygons on one image for binary segmentation
        :param polygons: list of polygons [x1, y1, x2, y2, ..., xn, yn]
        :param img_h: image height
        :param img_w: image width
        :return: image mask (h,w) with values {0,1} (1 where there is the object)
        """
        black_img = Image.new('L', (img_w, img_h), 0)
        for polygon in polygons:
            ImageDraw.Draw(black_img).polygon(polygon, outline=1, fill=1)
        mask = np.array(black_img)
        return mask


    def create_masks_multiclass(self, polygons1, polygons2, img_h, img_w):
        """
        create mask with polygons with 2 classes
        :param polygons1: list of polygons [x1, y1, x2, y2, ..., xn, yn] of class 1
        :param polygons2: list of polygons [x1, y1, x2, y2, ..., xn, yn] of class 2
        :param img_h: image height
        :param img_w: image width
        :return: image mask (h,w,3) with values color1 for class 1, color2 for class2 and black background
        """

        black_img = Image.new('RGB', (img_w, img_h), (0, 0, 0))  # black background
        for p1 in polygons1:
            ImageDraw.Draw(black_img).polygon(p1, outline=(self.color1[0],self.color1[1],self.color1[2]), fill=(self.color1[0],self.color1[1],self.color1[2]))
        for p2 in polygons2:
            ImageDraw.Draw(black_img).polygon(p2, outline=(self.color2[0],self.color2[1],self.color2[2]), fill=(self.color2[0],self.color2[1],self.color2[2]))
        mask = np.array(black_img)
        return mask

    # MODIFIED
    #################################################################################################
    def create_masks_one_hot(self, mask):
        """
        Create one-hot encoding masks for all masks
        :param masks: list of all the masks created with the function create_masks_multiclass
        :return: one-hot encoding masks with channel 0: background, 1: class 1, 2: class 2
        """
        labels_old = [[0, 0, 0], self.color1, self.color2]
        labels_new = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for i, label in enumerate(labels_old):
            mask[np.all(mask == label, axis=-1)] = labels_new[i]

        return mask
    #################################################################################################