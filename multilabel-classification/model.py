import torch.nn as nn
import torch.nn.functional as F

class MultiLabelImageClassifier(nn.Module):
    def __init__(self, num_classes=16):
        super(MultiLabelImageClassifier, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        
        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # fully connected layers
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)  # Adjusted input size
        self.fc2 = nn.Linear(1024, num_classes)

        # dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
                
        # flatten the input for the fully connected layers
        x = x.view(-1, 512 * 14 * 14)  # Adjusted view size
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits
