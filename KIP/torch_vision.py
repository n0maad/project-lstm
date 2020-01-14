'''
    PIL = Python Image Library

    ndarray refers to multidimensional arrays

    transforms.Compose() composes several transforms together.
    
    transfroms.ToTensor() converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        (H x W x C) is used by CNTK, whilst (C x H x W) is used by CUDA 
        H = height, W = width, C = colour index/channel
            For more info https://docs.microsoft.com/en-us/cognitive-toolkit/Archive/CNTK-Evaluate-Image-Transforms
        Returns a converted image of type tensor
            
    transforms.Normalize() normalizes a tensor image with mean and standard deviation.
        for n channels, this transform will normalize each channel of the input torch.*Tensor i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]
        paramters include:
            mean (sequence) – Sequence of means for each channel.
            std (sequence) – Sequence of standard deviations for each channel.
            inplace (bool,optional) – Bool to make this operation in-place.
        returns a normalised tensor image of type tensor


'''

import torch
import torchvision                                  # popular datasets for computer vision.
import torchvision.transforms as transforms         # library of image transformations.
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

# Composes several images together.
transform_data = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])    # Transforms the tensor to CHW (for CUDA use) and normalises the values.

# mkdir/store downloaded data
# obtain the training/testing dataset (boolean - True = train, False = test)
# download the data
# transform(s) applied to the data using 'transform_data'
trainset = torchvision.datasets.CIFAR10(root='../Data/CIFAR10_data', train = True, download = True, transform = transform_data)
# this returns a tuple (image, target[index])

# determine lenght of training dataset
print(len(trainset))

# show images using matplotlib.pyplot
plt.imshow(trainset.data[1])
#plt.show()
#print(trainset.train_label[1])             # this argument doesn't exist.


batch_size = 10

'''
Dataloader prepares the input data for training by creating a way to iterate over indices of dataset elements.
batch_size refers to how many elements are in each batch.
shuffle = True shuffles the data around each epoch.
num_workers refers to how many subprocesses to use for dataloading. 
    0 means the data will be loaded in the main process.
    2 means the data is loaded in parallel.
'''
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 2)    

for i, data in enumerate(trainloader):
    data, labels = data

    print('Iteration ', i)
    print('')
    print('type(data): ', type(data))
    print('data.size(): ', data.size())
    print('')
    print('type(labels): ', type(labels))
    print('labels.size(): ', labels.size())

    # Model training happens here

    break