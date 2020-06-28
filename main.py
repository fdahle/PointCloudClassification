import numpy as np

from torch.utils.data import DataLoader

from data import *
from train import *

#data = collectData("dataset/processed", k=1024)
train_ds = readData("dataset/processed/trainData.pickle")
test_ds = readData("dataset/processed/testData.pickle")

train_set = PointCloudData(train_ds)
test_set = PointCloudData(test_ds)

inv_classes = {i: cat for cat, i in train_set.classes.items()};

print('Sample pointcloud shape: ', train_set[0]['pointcloud'].size())

train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=test_set, batch_size=64)

train(100, train_loader, valid_loader, save=False)
