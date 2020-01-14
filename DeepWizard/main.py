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
        defines and initialises the weights, biases and caluclates input data @ (dot product) weights + bias, for that specific linear layer.
        Parameters:
            in_features - size of each input sample
            out_features - size of each output sample
            bias - will learn an additive bias, if set to True.

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

# Step 3 - Create Class Model (LSTM - single layer)
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()        # same as typing super().__init__():
        self.hidden_dim = hidden_dim        # hidden dimensions per layer
        self.layer_dim = layer_dim          # number of hidden layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True)   # batch_first, if True, provide the input and output tensors as (batch, seq, feature)
                                                                                    # batch_first=False will use the first dimension as the sequence dimension.
                                                                                    # batch_first=True will use the second dimension as the sequence dimension.
        # Readout layer??
        self.fc = nn.Linear(hidden_dim, output_dim)                                 # Applies a linear transformation to the date - y = xA^t + b
                                                                                    # ??hidden_dim are used because nn.Linear(in_features) refers to the size of each input sample for each layer not just the input layer.
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
        out = self.fc(out[:, -1, :])                                                # Each section refers each dimension. The last/bottom "slice" of the cube.
        
        # out.size() --> 100, 10
        return out

# Step 4: Instantiate Model Class
input_dim   = 28
hidden_dim  = 100
layer_dim   = 1         # no. of hidden layers
output_dim  = 10

lstm_model = LSTM(input_dim, hidden_dim, layer_dim, output_dim)

# Step 5 - Instantiate Loss Class (Classification Problem)
# Given an input and a target, a criterion computes a gradient according to a given loss function
criterion = nn.CrossEntropyLoss()   

# Step 6 - Instantiate Optimizer Class
learning_rate = 0.1
optimiser = torch.optim.SGD(lstm_model.parameters(), lr = learning_rate)
'''
    torch.optim.SGD implements stochastic gradient descent (optionally with momentum)
    it requires the parameters of the model (model.parameters()) and learning rate.
    optional arguments include a momentum factor, weight_decay, dampening and nesterov momentrum.

    torch.optim.SGD returns a scalar value

'''

'''
This model will contain six parameters, which include:
    Input to Hidden Layer Affine Function
    Hidden layer to output affine function
    Hidden layer to hidden layer affine function

An affine function consists of a linear function + a constant {f(x,y,z) = Ax + By + Cz + D}

'''
for i in range(len(list(lstm_model.parameters()))):
    print(list(lstm_model.parameters())[i].size())

'''
Input data ---> Gates
torch.Size([400, 28])   refers to the weight values
torch.Size([400])       refers to the bias values

Hidden States ---> Gates
torch.Size([400, 100])  refers to the weight values
torch.Size([400])       refers to the bias values

Hidden State ---> Output
torch.Size([10, 100])   refers to the weight
torch.Size([10])        refers to the bias
'''

# Step 7 - Train the model
'''
    Convert the inputs/labels into variables
    Clear gradient buffets
    Get output based on input values
    Get loss value
    Get gradients w.r.t (with respect to) parameters
    Update parameters using gradients
'''
# Single hidden layer

seq_dim = 28    # no. of steps to unroll

iter = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, seq_dim, input_dim).requires_grad_()   # Load images as a torch tensor with gradient accumulation abilities
        optimiser.zero_grad()                                           # Clears gradient w.r.t. parameters
        outputs = lstm_model(images)                                    # Forward pass to obtain outputs | output.size() = [100, 10]
        loss = criterion(outputs, labels)                               # Calculates Loss: softmax ---> cross entropy loss
        loss.backward()                                                 # Obtaining gradients w.r.t. parameters
        optimiser.step()                                                # Updating parameters

        iter += 1

        if iter % 500 ==0:
            # Calculate accuracy
            correct = 0
            total   = 0

            # Iterate through test dataset
            for images, labels in test_loader:
                images = images.view(-1, seq_dim, input_dim)    # Resizes images
                outputs = lstm_model(images)                    # Forward pass only to obtain output
                _, predicted = torch.max(outputs.data, 1)       # Get predictions from the maximum value
                total += labels.size(0)                         # Total number of labels
                correct += (predicted == labels).sum()          # Total correct predictions
            
            accuracy = 100 * correct / total
            print('Iterations: {}. Loss: {}. Accuracy {}'.format(iter, loss.item(), accuracy))


# Model B: 2 Hidden Layers
'''
    Unrolls 28 steps
        each step input size = 28 x 1
        total per unroll = 28 x 28
            feed forward nn input size = 28x28
    
'''
            