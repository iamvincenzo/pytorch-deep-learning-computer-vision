import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# BGR color encoding
COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": (0, 255, 0)},
    "index": {"ids": [0, 5, 6, 7, 8], "color": (255, 128, 0)},
    "middle": {"ids": [0, 9, 10, 11, 12], "color": (255, 0, 0)},
    "ring": {"ids": [0, 13, 14, 15, 16], "color": (0, 255, 255)},
    "little": {"ids": [0, 17, 18, 19, 20], "color": (0, 0, 255)},
}


def get_keypoint_location(heatmaps):
    """
    Calculate the (x, y) locations of keypoints based on heatmaps.

    Args:
        heatmaps (torch.Tensor): Tensor representing heatmaps for keypoints. Shape: (C x H x W).

    Returns:
        torch.Tensor: x locations of keypoints (1D tensor).
        torch.Tensor: y locations of keypoints (1D tensor).
    """
    N = heatmaps.shape[1]

    # step 1:
    # compute the sum of all heatmap entries for each heatmap of the channel (C x H x W) --> (21 x 1 x 1)
    sums = torch.sum(heatmaps, dim=(1, 2), keepdims=True)

    # step 2:
    # for each heatmap of the channel, divide all its entry by its corresponding sum of all heatmap entries (normalize)
    heatmaps /= sums

    # step 3:
    # for each heatmap of the channel, calculate sums along x (row) and y (column) axis. Result is two vector of size N (21 x N) --> (21 x N)
    col_sums = torch.sum(heatmaps, dim=1)
    row_sums = torch.sum(heatmaps, dim=2)

    # step 4:
    # to get (x, y) locations calculate dot product between column/row sums and index array [0,...,N]. (N x 1)
    indexes = torch.arange(N, dtype=torch.float32).reshape(N, 1)

    # x-(21 x 1) --> (21), y-(21 x 1) --> (21)
    x_locations = torch.matmul(col_sums, indexes).squeeze()
    y_locations = torch.matmul(row_sums, indexes).squeeze()

    return x_locations, y_locations


def plot_image(image, heatmaps, keypoints, resize=128):
    """
    Plot the original image, keypoints, and heatmaps.

    Args:
        image (torch.Tensor): Original image tensor.
        heatmaps (torch.Tensor): Tensor representing heatmaps for keypoints. Shape: (C x H x W).
        keypoints (list): List of (x, y) coordinates for keypoints.
        resize (int, optional): Size for resizing. Default is 128.
    """
    # convert tensors to numpy arrays for visualization
    image = (image.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(dtype=np.uint8)
    heatmaps = (heatmaps.squeeze().cpu().numpy() * 255.0).astype(dtype=np.uint8)

    # highlight keypoints on the image.
    for u, v in keypoints:
        u, v = int(u * resize), int(v * resize)
        if (0 <= u < resize) and (0 <= v < resize):
            image[v, u, :] = [255, 0, 0]

    # calculate the sum of heatmaps for visualization
    htot = np.sum(heatmaps, axis=0)

    # plot the original image, keypoints, and heatmaps
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(image), plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(htot.squeeze()) # , cmap="gray")
    plt.title("Heatmap")

    plt.show()


def draw_keypoints_connection(image, uv_coords):
    """
    Draw connections between keypoints on the given image.

    Args:
        image (torch.Tensor): Original image tensor.
        uv_coords (tuple): Tuple containing two 1D tensors representing x and y coordinates of keypoints.

    Returns:
        None: Displays the image with keypoints connections.
    """
    # convert the image tensor to a numpy array for visualization
    image = (image.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # unpack uv_coords tuple
    u_coords, v_coords = uv_coords

    # loop over different keypoint maps and draw connections
    for _, cmap in COLORMAP.items():
        for k in range(len(cmap["ids"])):
            if k < len(cmap["ids"]) - 1:
                # get indices for the current connection
                idx, idx1 = cmap["ids"][k], cmap["ids"][k + 1]

                # extract start and end points for the line
                start_point = int(u_coords[idx]), int(v_coords[idx])
                end_point = int(u_coords[idx1]), int(v_coords[idx1])

                radius = 1
                thickness = 1
                color = cmap["color"]

                # draw the line on the image
                image = cv2.line(image, start_point, end_point, color, thickness)

                # draw circular points at junctions
                image = cv2.circle(image, start_point, radius, color, -1)
                image = cv2.circle(image, end_point, radius, color, -1)

    # display the image with keypoints connections (BGR to RGB for matplotlib)
    plt.imshow(image[:, :, ::-1])
    plt.show()
