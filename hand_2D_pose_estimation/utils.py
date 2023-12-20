import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


def get_keypoint_location(heatmaps):
    N = heatmaps.shape[2]

    # step 1: compute the sum of all heatmap entries
    # for each heatmap of the batch (B x C x H x W) --> (B x 1 x 1 x 1)
    sums = torch.sum(heatmaps, dim=(2, 3), keepdims=True)

    # print(sums, "\n", sums.shape)

    # step 2: divide each entry by sum of all heatmap entries. Now all entries sum up to 1.
    # for each heatmap of the batch (B x C x H x W) --> (B x 1 x N x N)
    heatmaps /= sums

    # print(heatmaps, "\n", heatmaps.shape)

    # step 3: calculate sums along x (row) and y (column) axis. Result is two vector of size N.
    # (for each heatmap of the batch) - (B x 1 x N) --> (B x 1 x N)
    col_sums = torch.sum(heatmaps, dim=2)
    row_sums = torch.sum(heatmaps, dim=3)

    # print(col_sums, "\n", col_sums.shape)
    # print(row_sums, "\n", row_sums.shape)

    # step 4: to get (x, y) calculate dot product between column/row sums and index array [0,...,N].
    indexes = torch.arange(N, dtype=torch.float32).reshape(N, 1)

    # print(indexes, "\n", indexes.shape)

    # x-(B x 1 x 1), y-(B x 1 x 1)
    x_locations = torch.matmul(col_sums, indexes)
    y_locations = torch.matmul(row_sums, indexes)

    # print(x_locations, "\n", x_locations.shape)
    # print(y_locations, "\n", y_locations.shape)

    return x_locations, y_locations


def plot_image(image, heatmaps, keypoints, resize=128):
    image = (image.cpu().numpy().transpose(1, 2, 0) * 255.).astype(dtype=np.uint8)
    heatmaps = (heatmaps.squeeze().cpu().numpy() * 255.).astype(dtype=np.uint8)

    for u, v in keypoints:
        u, v = int(u * resize), int(v * resize)
        if (0 <= u < resize) and (0 <= v < resize):
            image[v, u, :] = [255, 0, 0]
            
    htot = np.sum(heatmaps, axis=0)            
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1); plt.imshow(image), plt.title("Image")
    plt.subplot(1, 2, 2); plt.imshow(htot.squeeze(), cmap="gray"); plt.title("Heatmap")
    
    plt.show()
