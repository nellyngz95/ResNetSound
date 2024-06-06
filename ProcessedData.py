import torch.nn as nn
from DatasetRN import MyDataset
import pathlib
import yaml
import torch
from pathlib import Path

# Define the processed directory
processed_dir = Path('/Users/nellygarcia/Documents/InformationRetrivalPhd/Dataset/Processed')

# Create dataset instances
train_dataset = MyDataset(cfg, train_files, train_labels, processed_dir)
valid_dataset = MyDataset(cfg, val_files, val_labels, processed_dir)
test_dataset = MyDataset(cfg, test_files, test_labels, processed_dir)

BS = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BS, shuffle=True, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BS, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BS, pin_memory=True)

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.pooling = nn.AdaptiveAvgPool2d((8, 8)) # extended
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.pooling(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x