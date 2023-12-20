"""
Credits:
    https://towardsdatascience.com/gentle-introduction-to-2d-hand-pose-estimation-approach-explained-4348d6d79b11
    https://towardsdatascience.com/gentle-introduction-to-2d-hand-pose-estimation-lets-code-it-6c82046d4acf
"""
import os
import cv2
import json
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms as T 


def create_dataframe(root):
    """
    Create a Pandas DataFrame from FreiHAND dataset.

    Args:
        - root (str): Root directory containing FreiHAND dataset.

    Returns:
        - pd.DataFrame: DataFrame containing file paths, K_matrix, and xyz coordinates.
    """
    all_images = np.sort(glob(os.path.join(root, "training/rgb/*.jpg")))  # get sorted list of image file paths
    all_images = all_images[:32560]  # limit the number of images

    df = pd.DataFrame(all_images, columns=["filepath"])  # create a DataFrame with file paths

    with open(os.path.join(root, "training_K.json"), "r") as f:
        K_matrix = np.array(json.load(f))  # load and convert camera matrix (K) from JSON file

    df["K_matrix"] = list(K_matrix)  # add K_matrix column to DataFrame

    with open(os.path.join(root, "training_xyz.json"), "r") as f:
        xyz = np.array(json.load(f))  # load and convert 3D coordinates (xyz) from JSON file

    df["xyz"] = list(xyz)  # add xyz column to DataFrame

    return df


# Custom dataset class for FreiHAND
class FreiHandDataset(Dataset):
    def __init__(self, df, resize, n_keypoints):
        super(FreiHandDataset, self).__init__()
        """
        A PyTorch Dataset class for loading FreiHAND images and masks.
        https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html

        Args:
            - df (pd.DataFrame): DataFrame containing file paths, K_matrix, and xyz coordinates.
            - resize (int): Target size for resizing images.
            - n_keypoints (int): Number of keypoints.

        Returns:
            - None
        """
        self.df = df
        self.resize = resize
        self.n_keypoints = n_keypoints

        # define image transformation pipeline
        self.transform = T.Compose([
            T.ToPILImage(),    # convert image to PIL format
            T.Resize(size=(resize, resize)),  # resize image
            ToTensor()  # convert image to PyTorch tensor
        ])

    def project_points(self, xyz, K):
        """
        Project 3D coordinates onto 2D plane using camera matrix K.

        Args:
            - xyz (np.array): 3D coordinates.
            - K (np.array): Camera matrix.

        Returns:
            - np.array: 2D coordinates.
        """
        xyz = np.array(xyz)     # convert xyz to NumPy array
        K = np.array(K)         # convert K to NumPy array
        uv = np.matmul(K, xyz.T).T  # perform matrix multiplication to get 2D coordinates

        return uv[:, :2] / uv[:, -1:]  # normalize and return 2D coordinates

    def generate_heatmaps(self, keypoints):
        """
        Generate heatmaps from 2D keypoints.

        Args:
            - keypoints (list): List of 2D keypoints.

        Returns:
            - np.array: Heatmaps.
        """
        heatmaps = np.zeros([self.n_keypoints, self.resize, self.resize])  # initialize empty heatmaps

        for k, (u, v) in enumerate(keypoints):
            u, v = int(u * self.resize), int(v * self.resize)  # scale (denormalize) and convert to integer
            if (0 <= u < self.resize) and (0 <= v < self.resize):
                heatmaps[k, int(v), int(u)] = 1  # set the corresponding pixel to 1 if within bounds

        return heatmaps

    def blur_heatmpas(self, heatmaps):
        """
        Apply Gaussian blur to heatmaps.

        Args:
            - heatmaps (np.array): Input heatmaps.

        Returns:
            - np.array: Blurred heatmaps.
        """
        for i in range(len(heatmaps)):
            heatmaps[i] = cv2.GaussianBlur(heatmaps[i], (51, 51), 3)  # apply Gaussian blur
            heatmaps[i] = heatmaps[i] / heatmaps[i].max()  # normalize to values between 0 and 1

        return heatmaps

    def __getitem__(self, index):
        """
        Retrieves and transforms image, heatmaps, and keypoints at the specified index.

        Args:
            - index (int): Index of the dataset.

        Returns:
            - dict: Dictionary containing transformed image, heatmaps, and keypoints.
        """
        image = cv2.imread(self.df["filepath"].iloc[index], cv2.IMREAD_COLOR)  # read image in BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        K_matrix = self.df["K_matrix"].iloc[index]  # get camera matrix (K) for the current image
        xyz = self.df["xyz"].iloc[index]  # get 3D coordinates for the current image

        # get and normalize keypoints
        keypoints = self.project_points(xyz=xyz, K=K_matrix) / image.shape[1]
        heatmaps = self.generate_heatmaps(keypoints=keypoints)
        heatmaps = self.blur_heatmpas(heatmaps=heatmaps)

        # apply image transformation
        image = self.transform(image)
        heatmaps = T.ToTensor()(heatmaps.transpose(1, 2, 0).astype(dtype=np.float32))

        return {"image": image, "heatmaps": heatmaps, "keypoints": keypoints}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            - int: Total number of samples in the dataset.
        """
        return len(self.df)
