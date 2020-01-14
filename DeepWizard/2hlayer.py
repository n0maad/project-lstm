'''
    LSTM w/ 2 hidden layers
           
    Steps involved:
        Step 1: Load Dataset
        Step 2: Make Dataset iterable
        Step 3: Create Model Class
        Step 4: Instantiate Model Class
        Step 5: Instantiate Loss Class
        Step 6: Instantiate Optimiser Class
        Step 7: Train Model
'''

import torch
import torch.nn as nn

import torchvision.transforms as tranforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Step 1: Load Dataset
train_dataset = datasets.MNIST(root='./data', train = True, transform = tranforms.ToTensor(), download = True)
test_datasets   = datasets.MNIST(root='./data', train = False, transform = transforms.ToTensor())

#Step 2: Make Dataset iterable
batch_size = 100
n_iters = 3000

num_epochs = n_iters / (len(train_dataset) / )
