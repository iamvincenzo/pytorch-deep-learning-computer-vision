import torch
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


def show_preds(images: torch.Tensor, masks: torch.Tensor, preds: torch.Tensor, alpha: Optional[float] = None) -> None:
    """
    Display a batch of images along with their ground truth masks and predicted masks.

    Parameters:
        - images (torch.Tensor): Tensor of input images.
        - masks (torch.Tensor): Tensor of ground truth masks.
        - preds (torch.Tensor): Tensor of predicted masks.
        - alpha (float, optional): Transparency level for overlaying masks on images.
          If None, the default value is 0.6.
    
    Returns:
        - None.
    """
    for image, mask, pred in zip(images, masks, preds):
        # define a figure
        plt.figure(figsize=(12, 12))

        # set transparency for overlaying ground truth mask and prediction on the image
        alpha = 0.6 if alpha is None else alpha

        # convert the mask values to a color map
        cmap = plt.get_cmap('viridis')

        # original image
        np_img = (image.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(dtype=np.uint8)    
        # normalize original image
        np_img_normalized = np_img / 255.0  # assuming pixel values are in the range [0, 255]    
    
        # ground truth mask
        np_gt_msk = (mask.squeeze().cpu().numpy()).astype(dtype=np.uint8)
        # apply the colormap to the ground truth mask
        colored_gt_mask = cmap(np_gt_msk / np.max(np_gt_msk)) if np.max(np_gt_msk) != 0 else cmap(np.zeros_like(np_gt_msk))  # handle division by zero
        # blend the original image with the colored ground truth mask
        blended_img_gt_mask = alpha * np_img_normalized + (1 - alpha) * colored_gt_mask[..., :3]  # use only RGB channels
        
        # predicted mask
        np_pred_msk = (pred.squeeze().cpu().numpy()).astype(dtype=np.uint8)
        # convert the predicted mask values to a color map
        colored_pred_mask = cmap(np_pred_msk / np.max(np_pred_msk)) if np.max(np_pred_msk) != 0 else cmap(np.zeros_like(np_pred_msk))  # handle division by zero        
        # blend the original image with the colored predicted mask
        blended_img_pred = alpha * np_img_normalized + (1 - alpha) * colored_pred_mask[..., :3]  # use only RGB channels

        # print(f"gt-mask: {np.unique(np_gt_msk)}, pred-mask: {np.unique(np_pred_msk)}")

        # display all
        plt.subplot(2, 3, 1); plt.imshow(np_img); plt.title("Image")
        plt.subplot(2, 3, 2); plt.imshow(blended_img_gt_mask); plt.title("Mask")
        plt.subplot(2, 3, 3); plt.imshow(blended_img_pred); plt.title("Prediction mask")
        plt.subplot(2, 3, 4); plt.imshow(np_img); plt.title("Image")
        plt.subplot(2, 3, 5); plt.imshow(np_gt_msk); plt.title("Mask")
        plt.subplot(2, 3, 6); plt.imshow(np_pred_msk); plt.title("Prediction mask")
        plt.show(block=False); plt.pause(7); plt.close()


# def show_batch_preds(images, masks, preds):
#     """
#     Display a batch of images along with their ground truth masks and predicted masks.

#     Parameters:
#         - images (torch.Tensor): Tensor of input images.
#         - masks (torch.Tensor): Tensor of ground truth masks.
#         - preds (torch.Tensor): Tensor of predicted masks.
    
#     Returns:
#         - None
#     """
#     for image, mask, pred in zip(images, masks, preds):
#         # define a figure
#         plt.figure(figsize=(12, 12))

#         # original image
#         np_img = (image.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(dtype=np.uint8)        
        
#         # ground truth mask
#         np_gt_msk = (mask.squeeze().cpu().numpy()).astype(dtype=np.uint8)        
        
#         # predicted mask
#         np_pred_msk = (pred.squeeze().cpu().numpy()).astype(dtype=np.uint8)
        
#         # display all
#         plt.subplot(1, 3, 1); plt.imshow(np_img); plt.title("Image")
#         plt.subplot(1, 3, 2); plt.imshow(np_gt_msk); plt.title("Mask")
#         plt.subplot(1, 3, 3); plt.imshow(np_pred_msk); plt.title("Prediction mask")
#         plt.show(block=False); plt.pause(5); plt.close()

# def show_batch_preds_with_transparency(images, masks, preds, alpha=None):
#     """
#     Display a batch of images along with their ground truth masks and predicted masks.

#     Parameters:
#         - images (torch.Tensor): Tensor of input images.
#         - masks (torch.Tensor): Tensor of ground truth masks.
#         - preds (torch.Tensor): Tensor of predicted masks.
#         - alpha (float, optional): Transparency level for overlaying masks on images.
#           If None, the default value is 0.6.
    
#     Returns:
#         - None
#     """
#     for image, mask, pred in zip(images, masks, preds):
#         # define a figure
#         plt.figure(figsize=(12, 12))

#         # set transparency for overlaying ground truth mask and prediction on the image
#         alpha = 0.6 if alpha is None else alpha

#         # convert the mask values to a color map
#         cmap = plt.get_cmap('viridis')

#         # original image
#         np_img = (image.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(dtype=np.uint8)    
#         # normalize original image
#         np_img_normalized = np_img / 255.0  # assuming pixel values are in the range [0, 255]    
    
#         # ground truth mask
#         np_gt_msk = (mask.squeeze().cpu().numpy()).astype(dtype=np.uint8)
#         # apply the colormap to the ground truth mask
#         colored_gt_mask = cmap(np_gt_msk / np.max(np_gt_msk)) if np.max(np_gt_msk) != 0 else cmap(np.zeros_like(np_gt_msk))  # handle division by zero
#         # blend the original image with the colored ground truth mask
#         blended_img_gt_mask = alpha * np_img_normalized + (1 - alpha) * colored_gt_mask[..., :3]  # use only RGB channels
        
#         # predicted mask
#         np_pred_msk = (pred.squeeze().cpu().numpy()).astype(dtype=np.uint8)
#         # convert the predicted mask values to a color map
#         colored_pred_mask = cmap(np_pred_msk / np.max(np_pred_msk)) if np.max(np_pred_msk) != 0 else cmap(np.zeros_like(np_pred_msk))  # handle division by zero
        
#         print(np_img_normalized.shape, colored_pred_mask.shape, np.zeros_like(np_img).shape)
        
#         # blend the original image with the colored predicted mask
#         blended_img_pred = alpha * np_img_normalized + (1 - alpha) * colored_pred_mask[..., :3]  # use only RGB channels

#         # display all
#         plt.subplot(1, 3, 1); plt.imshow(np_img); plt.title("Image")
#         plt.subplot(1, 3, 2); plt.imshow(blended_img_gt_mask); plt.title("Mask")
#         plt.subplot(1, 3, 3); plt.imshow(blended_img_pred); plt.title("Prediction mask")
#         plt.show(block=False); plt.pause(5); plt.close()
