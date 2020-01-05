'''
    Deep learning wizard LSTM tutorial.

    Nathan Thomas - 03/01/2020.

    torch.nn.functional 
        contains all the functions (activation, loss, etc) available in the torch.nn library as well as a wider range of loss and activation functions.

    torch.nn.Module 
        is a base class for all neural network modules.
        behaves like a function (is callable).
        contains states (nn layer weights).
        knows the contained parameters and cal zero all gradients and loop through them for weight updates.

    torch.nn.Parameter 
        is a kind of tensor that is to be considered a module parameter. 
        When they are assigned as Module attributes they are automatically added to the list of its parameters.
        wrapper for a tensor telling nn.Module that weights need to be updated during backprop (only tensors with requires_grad are updated).

    torch.nn.Linear 
        defines and initialises the weights, biases and caluclates input data @ (dot product) weights + bias, for a linear layer.

    torch.nn.Sequential
        A sequential object runs each of the modules contained within it, in a sequential manner.
    
    torch.optim 
        allows for the use of various optimisation algorithms, which update the weights of nn.Parameters during backprop (backwards step).

    torch.utils.data.TensorDataset 
        is a Dataset wrapping tensors. By defining the length and wat of indexing it provides an avenue to iterate, 
        index and slice along the first dimension of a tensor.

    torch.utils.data.DataLoader 
        is responsible for managing batches. DataLoaders can be made from any Dataset and it makes it easier to iterate over the batches.
        takes any TensorDataset and creates an iterator which returns batches of data.

    torchvision refers to packages that consist of popular datasets, model architectures and image transformations
    
    before training model.train() is called and model.eval() is called before inference (a reached conclusion based on evidence and reasoning),
    as these are used by layers nn.BatchNorm2d and nn.Dropout to ensure appropriate behavious for these different phases.
'''

import torch
import torch.nn as nn

import torchvision.transforms as transforms  
import torchvision.datasets as datasets

from torch.utils.data import DataLoader 
from torch.utils.data import TensorDataset


# Step 1 - Loading MNIST Train Datasets
train_datasets  = datasets.MNIST(root='./data', train = True, transform = transforms.ToTensor(), download = True)
test_datasets   = datasets.MNIST(root='./data', train = False, transform = transforms.ToTensor())

print(train_datasets.train_data.size())
print(train_datasets.train_labels.size())

print(test_datasets.train_data.size())
print(test_datasets.train_labels.size())

# Step 2 - Make Datasets iterable
batch_size  = 100
n_iters     = 3000

#3000/(6000/100)
num_epochs = n_iters / (len(train_datasets)/batch_size)
num_epochs = int(num_epochs)

#DataLoader sets up minibaches automatically for easy iterations over a dataset.
train_loader = DataLoader(train_datasets, batch_size, True)
test_loader  = DataLoader(test_datasets, batch_size, False)


