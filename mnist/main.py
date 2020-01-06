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

        An epoch is when the ENTIRE dataset has performed a single foward and backwards rotation.
        Batch size refers to the total number of inputs present in a single batch.
            batch size and number of batches are two different things
                number of batches refers to how many segments the ENTIRE dataset is divided into.
                batch size refers to the number of inputs each batch contains.
        An iteration is the number of batches required to complete one epoch

        The number of iterations equals the number of batches for one epoch
            e.g. a dataset contains 100 datapoints. Dividing it into 5 batches means that the "process" needs to be iterated 5 times (to accomodate said 5 batches).
                 each batch will have a batch size of 20 datapoints.
                 i.e. iterations, no. of batches = 5
                      batch size = 20
                      epoch = 1


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

num_epochs = n_iters / (len(train_datasets)/batch_size)
num_epochs = int(num_epochs)                                # 5 epochs

train_loader = DataLoader(train_datasets, batch_size, True) # DataLoader sets up minibaches automatically for easy iterations over a dataset.
test_loader  = DataLoader(test_datasets, batch_size, False)

# Step 3 - Create Class Model (LSTM)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()        # same as typing super().__init__():
        self.hidden_dim = hidden_dim        # hidden dimensions
        self.layer_dim = layer_dim          # number of hidden layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True)   # batch_first, if True, provide the input and output tensors as (batch, seq, feature)
                                                                                    # batch_first=False will use the first dimension as the sequence dimension.
                                                                                    # batch_first=True will use the second dimension as the sequence dimension.
        # Readout layer??
        self.fc = nn.Linear(hidden_dim, output_dim)                                 # Applies a linear transformation to the date - y = xA^t + b
                                                                                    # hidden_dim are used because nn.Linear(in_features) refers to the size of each input sample for each layer not just the input layer.
    def forward(self, x):
        # Initialise hidden states with zeroes
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialise the cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))                    # detach() ensures the gradient is not backpropagated.
                                                                                    # We need to detach as we are doing truncated backpropagation through time (BPTT)
                                                                                    # If we don't, we'll backprop all the way to the start even after going through another batch
        '''
            nn.LSTM returns:
                output = a tensor containing the output features of the final layer of the LSTM.
                         the shape of this output is (seq_len, batch, num_directions * hidden_dim).
                hn = a tensor containing the hidden state for t = seq_len
                cn = a tensor containing the cell state for t = seq_len
                     both hn and cn can be seperated into their respective layers using hn/cn.view(num_layers, num_directions, batch, hidden_size)
        '''
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out



