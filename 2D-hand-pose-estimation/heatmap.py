import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torchvision.transforms import transforms as T

# tenros: (B x C x H x W)
# numpy: (H x W x C)

def create_circle_mask(size, radius, center):
    y, x = np.ogrid[:size, :size]
    mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2) <= radius ** 2
    
    return mask.astype(float)

def create_heatmap_batch(channels, height, width, circle_radius):
    heatmaps = np.zeros((channels, height, width), dtype=float)
    
    for i in range(channels):
        # Randomly select position for the circle
        center = np.random.randint(circle_radius, height - circle_radius, size=2)
        circle_mask = create_circle_mask(height, circle_radius, center)
        heatmaps[i, :, :] = circle_mask
    
    return heatmaps

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


if __name__ == "__main__":
    # parameters
    batch_size = 21
    height, width = 128, 128
    circle_radius = 7

    # create the heatmap batch
    heatmap_batch = create_heatmap_batch(batch_size, height, width, circle_radius)

    heatmap_batch = heatmap_batch.astype(dtype=np.float32)

    h_t = T.ToTensor()(heatmap_batch.transpose(1, 2, 0)).unsqueeze(0)

    x_locations, y_locations = get_keypoint_location(heatmaps=h_t)

    x_locations, y_locations = x_locations.squeeze(), y_locations.squeeze()

    for x, y in zip(x_locations, y_locations):
        print(f"Estimated (x, y) location: ({int(x.item())}, {int(y.item())})")

    # visualize the heatmaps and keypoints
    for heatmap, x, y in zip(heatmap_batch, x_locations, y_locations):
        plt.imshow(heatmap.squeeze(), cmap="gray")
        plt.scatter(int(x.item()), int(y.item()), c="red", marker="x")
        plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# heatmaps = np.array(
#     [
#         [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           [0., 1.0, 1.0, 1.0, 0.0, 0.0],
#           [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
#           [0.0, 1., 1.0, 1.0, 0.0, 0.0],
#           [0., 0.0, 0.0, 0.0, 0.0, 0.0]]],

#         [[[0., 0.0, 0.0, 0.0, 0.0, 0.0],
#           [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
#           [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
#           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           [0., 0.0, 0.0, 0.0, 0.0, 0.0]]],
#     ]
# ).astype(dtype=np.float32)

# sums = np.sum(heatmaps, axis=(2, 3), keepdims=True)

# # print(sums, sums.shape)

# heatmaps /= sums

# print(heatmaps)

# print()

# for heatmap in heatmaps:
#     plt.imshow(heatmap.squeeze(), cmap="gray")
#     plt.show()

# col_sums = np.sum(heatmaps, axis=2)
# row_sums = np.sum(heatmaps, axis=3)

# print(row_sums, row_sums.shape)

# print(col_sums, col_sums.shape)

# # Step 4: Calculate (x, y) location
# x_location = np.dot(col_sums, np.arange(6))
# y_location = np.dot(row_sums, np.arange(6))

# print(f"Estimated (x, y) location: ({x_location}, {y_location})")

# # Optionally, visualize the heatmaps and keypoints
# for heatmap, x, y in zip(heatmaps, x_location, y_location):
#     plt.imshow(np.squeeze(heatmap), cmap="gray")
#     plt.scatter(np.round(x.flatten()), np.round(y.flatten()), c='red', marker='x')
#     plt.show()
