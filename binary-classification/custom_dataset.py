import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CleanDirtyRoadDataset(Dataset):
    """
    """
    def __init__(self, X, y, resize, data_root, transform):
        """
        Custom dataset for loading 2D images.
        https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification/

        Args:
            X (DataFrame):
            y (DataFrame):
            resize (int):
            data_root (str): Root directory of the dataset.
            transform (callable): Optional transform to be applied on a sample.
        """
        super(CleanDirtyRoadDataset, self).__init__()
        self.X = X
        self.y = y
        self.data_root = data_root

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(resize, resize)), 
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        """
        Retrieves a specific item form the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the 2D image data and its corresponding class.
        """
        img_name = self.X.iloc[index]
        img_label = self.y.iloc[index]

        img_pth = self.data_root / img_name

        img = Image.open(img_pth).convert("RGB")

        img_tensor = self.transform(img)
        label_tensor = torch.tensor(img_label, dtype=torch.float32)

        return img_tensor, label_tensor

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.X)
