# TO CUSTOMIEZE
########################################################################################################################
# import cv2
# import numpy as np
# from glob import glob
# import albumentations as A
# # import matplotlib.pyplot as plt
# from torch.utils.data import Dataset
# from albumentations.pytorch.transforms import ToTensorV2


# class MRIDataset(Dataset):
#     def __init__(self, images, resize_h, resize_w, data_aug):
#         """
#         A PyTorch Dataset class for loading and augmenting MRI images and masks.
#         https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

#         Args:
#             - images (list): List of file paths for MRI images.
#             - data_aug (bool): Flag indicating whether to apply data augmentation.

#         Returns:
#             - None
#         """
#         super(MRIDataset, self).__init__()
#         self.resize_h = resize_h
#         self.resize_w = resize_w
#         self.all_images = images
#         self.all_masks = [fn[:-4] + "_mask" + fn[-4:] for fn in images]

#         # define image transformation pipeline
#         if data_aug:
#             self.transform = A.Compose([
#                 # A.Resize(height=self.resize_h, width=self.resize_w),
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.5),
#                 A.RandomRotate90(p=0.5),
#                 A.Rotate(limit=(-30, 30), border_mode=cv2.BORDER_CONSTANT, value=.0, p=0.5),
#                 A.RandomBrightnessContrast(p=0.2),
#                 A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
#                 A.Blur(blur_limit=(3, 7), p=0.5),
#                 # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#                 ToTensorV2()
#             ])
#         else:
#             self.transform = A.Compose([
#                 # A.Resize(height=self.resize_h, width=self.resize_w),
#                 ToTensorV2()
#             ])

#     def __getitem__(self, index):
#         """
#         Retrieves and transforms MRI image and mask at the specified index.

#         Args:
#             - index (int): Index of the dataset.

#         Returns:
#             - tuple: Tuple containing transformed MRI image and mask.
#         """
#         image = cv2.imread(self.all_images[index], cv2.IMREAD_UNCHANGED)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(self.all_masks[index], cv2.IMREAD_UNCHANGED)

#         image = image.astype(dtype=np.float32) / 255.0
#         mask = mask.astype(dtype=np.float32) / 255.0
#         transformed = self.transform(image=image, mask=mask)

#         # # DEBUG
#         # import matplotlib.pyplot as plt
#         # img = (transformed["image"].cpu().numpy().transpose(1, 2, 0) * 255).astype(dtype=np.uint8).copy()
#         # plt.imshow(img)
#         # plt.show()
#         # msk = (transformed["mask"].cpu().numpy() * 255).astype(dtype=np.uint8).copy()
#         # plt.imshow(msk, cmap="gray")
#         # plt.show()

#         return transformed["image"], transformed["mask"]

#     def __len__(self):
#         """
#         Returns the total number of samples in the dataset.

#         Returns:
#             - int: Total number of samples in the dataset.
#         """
#         return len(self.all_images)
########################################################################################################################
