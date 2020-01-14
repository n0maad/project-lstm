'''
    Introduction to neural networks

    This code will introduce:
        loss functions
        optimizers 
        training the nn
        saving and loading a trained nn
        training the nn on a GPU
    
    This will help provide an understanding of:
        neurons
        weights
        biases
        loss functions
        activation functions
        Dataloaders
        Preparing data for training
    
    This project will build a:
        milti-layer feedforward nn
'''

'''
    A neuron takes in various inputs (that are altered by a wieght) and sums them all together.
    Each neuron has a bias that is also added to the value of the accumulated inputs@wieghts.
    The output of each neuron is inputed into an activation function (sigmoid, tanh, ReLU) and produces an output, 
    which is used as the input for the next hidden layer.

    A neural network is simply an array of neurons, which make up a layer. Layers refer to the input layer, x*hidden layers and an output layer.
    Training a neural network is simply altering the weights and biases to produce the desired output.
'''

'''
    Activation functions are crucial to neural networks as they allow the network to learn complex boundaries.
    They bring in nonlinearality into the network as they decide whether the neuron should produce a value of 0 (neuron doesnt "fire") or x.
    Due to these activation functions the network can parition the data with complex boundaries.
    Signmoid (0,1)
    tanh(-1,1)
    ReLU = max(x,0) - most commonly used acitvation function. 
'''

import torch
import torch.nn as nn
import matplotlib.pyplot as pyplot

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

'''
    Creating custom datasets
'''

# Creating a pd dataframe
datafile = pd.read_csv('../Data/Iris/iris_data.csv')            # import data as pd.dataframe.
datafile = datafile.sample(frac=1).reset_index(drop=True)       # shuffle data and reset the index values.

# Divide data intro training(80%) and testing(20%).
train = datafile.loc[0:len(datafile)*0.8-1, :]           # creates df of training values.
test  = datafile.loc[len(datafile)*0.8:len(datafile), :]  # creates df of training labels.

# Seperate df into values and labels
train_labels = train.iloc[:,4]                                  # creates a df of just labels.
train_values = train.iloc[:,0:4]                                # creates a df of just values.

test_labels = test.iloc[:,4]
test_values = test.iloc[:,0:4] 

# Converts df to tensor
train_values = torch.Tensor(train_values.values)         
train_labels = torch.Tensor(train_labels.values)

test_values = torch.tensor(test_values.values, dtype = torch.float)           
test_labels = torch.tensor(test_labels.values, dtype = torch.long)

# Creates a TensorDataset
train_ds = TensorDataset(train_values,train_labels)             # Create dataset to be used with DataLoader.
test_ds  = TensorDataset(test_values,test_labels)

'''
    nn.Module is the base class for all network models
    All custom networks must inherit (be subclasses) from this class.
    They must include an initilise and forward function
'''

class irisNet(nn.Module):
    # defines all initial values within the class and creates the various layers and activation functions.
    # input_size = no. of attributes of the flower
    # hidden_layerX refers to the number of nodes per hidden layer
    # num_classes refer to the possible resulting classifications and number of neurons in the output layer
    def __init__(self, input_size, hidden_layer1, hidden_layer2, num_classes):          # dictates what happens per layer of the neural network.
        
        super(irisNet, self).__init__()
        # Linear's argumenents include the number of incoming and outgoing edges(nodes?) and generates a bais per edge
        self.fc1 = nn.Linear(input_size, hidden_layer1)     # ouputs results for hidden layer 1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)  # ouputs results for hidden layer 2
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_layer2, num_classes)    # provdies results for output layer

    # defines the computation performed at every call (iteration/epoch).
    # this function moves the results from one layer to the next - moving it foward.
    # batches are used, not the entire dataset. So each batch will produce is own classification.    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(x)
        out = self.relu2(out)
        return self.fc3(x)

# Create Dataloader
batch_size = 60

print('Instances in testing set', len(train_ds))
print('Instances in testing set', len(test_ds))

# allows for iteration through the dataset.
train_loader = DataLoader(train_ds, batch_size, True)
test_loader = DataLoader(test_ds, batch_size, True)

'''
    Loss functions and Optimisers

    Loss function (aka cost function) is an indicator of how well the system is doing in predicting the desired output.
    It functions as a feedback mechanism and helps fine tune the parameters (weights and biases) of the network.

    Once the loss has been calculated its gradient is determined w.r.t. to the models parameters (using back propagation).
    These gradients are inputed into the optimiser (of our choice), which then updates the model parameters accordingly.

    Optimisers simply updates tge models paramaters based on the gradients calculated from the loss.
'''

# Create nerual network model
model = irisNet(4, 100, 50, 3)
#print(model)    # prints out the values for each layer in the model

# Loss function
criterion = nn.CrossEntropyLoss()   # outputs a scalar

# SGD Optimiser (model parameters, learning rate)
learning_rate = 0.001
optimiser = torch.optim.SGD(model.parameters(), lr = learning_rate, nesterov=True, momentum=0.9)    #Nesterov uses "predictions" to ensure most efficient optimisation, which requires a momentum value (0.9 is std).

num_epochs = 500    # number of time to cycle through the entuire data set

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []


for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0

    for i, (items, classes) in enumerate(train_loader):     # cycle through one batch | items is a float tensor containing the attributes of the flower | classes is a long tensor containing the classes.
       
        # convert Tensor to Variable (why do we want to do that)
        items = Variable(items)
        classes = Variable(classes)

        model.train()                       # puts network into training mode

        optimiser.zero_grad()               # resets all the gradients to zero.
        outputs = model(items)              # performs a forward pass.
        loss = criterion(outputs)           # calculates the loss of the forward pass
        loss.backward()                     # calcualtes the gradients of the loss w.r.t. to parameters.
        optimiser.step()                    # updates the parameters based on the gradients

        # Record the correct predictions for training data
        train_total += classes.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == classes.data).sum()

        print('Epoch %d/%d, Iterations %d/%d, Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_ds)//batch_size, loss.data[0]))


    model.eval()    # put the model into evaluation mode

    # record the loss
    train_loss.append(loss.data[0])
    
    # training accuracy
    train_accuracy.append((100*train_correct/train_total))

    test_items = torch.Tensor(test_ds.data.values[:,0:4])
    test_classes = torch.LongTensor(test_ds.data.values[:,4])

    outputs = model(Variable(test_items))
    loss = criterion(outputs, Variable(test_classes))
    test_loss.append(loss.data[0])
    _,predicted = torch.max(outputs.data, 1)
    total = test_classes.size(0)
    correct = (predicted == test_classes).sum()
    test_accuracy.append((100*correct/total))

# Plot the loss of the model
fig = plt.figure(figsize=(12,8))
plt.plot(train_loss, label = 'train loss')
plt.plot(test_loss, label = 'test loss')
plt.title('Train and Test Loss')
plt.legend()
plt.show()

# Plot the accuracy of the model
fig = plt.figure(figsize=(12,8))
plt.plot(train_accuracy, label = 'train accuracy')
plt.plot(test_accuracy, label = 'accuracy loss')
plt.title('Train and Test Accuracy')
plt.legend()
plt.show()

