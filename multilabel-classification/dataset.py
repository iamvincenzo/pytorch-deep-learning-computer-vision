import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, skipcols, data_root, transform, resize):
        """
        Custom dataset for loading 2D images.
        https://www.kaggle.com/datasets/meherunnesashraboni/multi-label-image-classification-dataset

        Args:
            data_root (str): Root directory of the dataset.
            transform (callable): Optional transform to be applied on a sample.
            train (bool): Flag indicating whether to load training or testing data.
        """
        super(CustomImageDataset, self).__init__()
        self.df = dataframe
        self.skipcols = skipcols
        self.data_root = data_root
        self.transform = transform
        self.resize = resize

    def __getitem__(self, index):
        """
        Retrieves a specific item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the 3D point cloud data and its corresponding class index.
        """
        img_name = self.df.iloc[index]["Image_Name"]
        img_labels = self.df.iloc[index, self.skipcols:]
        img_pth = self.data_root / img_name
        # RGB prevent from grayscale images in the dataset
        img = Image.open(img_pth).convert("RGB")
        labels = torch.tensor(img_labels,
                              dtype=torch.float32)
        
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            transform = transforms.Compose([transforms.Resize(size=(self.resize, self.resize)),
                                            transforms.ToTensor()])   
            img_tensor = transform(img)
        
        return img_tensor, labels

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.df)