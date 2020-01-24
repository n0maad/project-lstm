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


        #requires_grad_() is a function that saves all gradients into a tensor for later use/pre processing?
'''

import torch
import torch.nn as nn

import torchvision.transforms as tranforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Allocating GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Load Dataset
train_dataset = datasets.MNIST(root='./data', train = True, transform = tranforms.ToTensor(), download = True)
test_dataset  = datasets.MNIST(root='./data', train = False, transform = tranforms.ToTensor())

# Step 2: Make Dataset iterable
batch_size = 100                                                # number of samples per batch.
n_iters = 3000                                                  # number of times to cycle through each batch.
num_epochs = int(n_iters / (len(train_dataset)/batch_size))     # number of cycles through the entire dataset.

train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
test_loader  = DataLoader(test_dataset, batch_size)

# Step 3: Create Model Class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, layer_dim, output_size, bidir, weight=0):
        super(LSTM, self).__init__()

        # Number of neurons in the hidden layer
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Determine if the LSTM is bidirectional
        self.bidir = bidir
    
        # Generates the LSTM aspect to the model
        self.lstm = nn.LSTM(input_size, hidden_dim, layer_dim, batch_first = True, bidirectional=self.bidir)      # batch_first = True makes the output a tensor(batch, seq(*), feature)
        ''' LSTM Output
        # output = (seq_len, batch, num_directions_hidden_size) from the last layer of the lstm
        # h_n = (num_layers * num_directions, batch, hidden_size) a tensor containing the hidden state for t = seq_len
        #   hidden states are the
        # c_n = (num_layers * num_directions, batch, hidden_size) a tensor containing the cell state for t = seq_len 
        #   cell states maintain the bulk of the networks information as it transfers relevant information throughout the process, which get altered by the various gates.
            This is how information from earlier 
        '''
        # Calculations performed in each layer
        if self.bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_size)
        else: self.fc = nn.Linear(hidden_dim, output_size)
        ''' Linear class
            
        '''
        if weight !=0:
            self.init_weights()

    def forward(self, x):
        
        if self.bidirectional:
            # Initialise the hidden states with zeroes and saves them in a tensor.
            h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_().to(device)
            # Initialise cell states
            c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_().to(device)
        else:
            # Initialise the hidden states with zeroes and saves them in a tensor.
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
            # Initialise cell states
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # Generate output, hidden and cell states
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))       # detach is required to ensure BPTT is done only in the current batch and not all batches.
        
        return self.fc(out[:,-1,:])                                    # only inputs the final layer of elements.
    '''
        weights are automatically assigned through nn.Linear() ---> self.weight = Parameter(torch.Tensor(out_features, in_features))
        Parameter is a class that assigns model attributes which adds them to the list of model parametes.
        A seperate function must be made to select a specific weight initialisation
    '''
    # Initialise weights
    def init_weights():
        int_range = 1
        self.weight.data.uniform_(-int_range, int_range)
        '''
            .data changes the values of the tensor
            .uniform_(from, to) fills the tensor with numbers from a continuous uniform distribution (1/(to - from))
        '''
        return 0

# Step 4: Instantiate Model Class

# hyper-parameters
input_size  = 28                # number of expected features in the input
hidden_dim  = 100               # number of features in the hidden state
layer_dim   = 2                 # number of recurrent layers
output_size = 10                # number of classes (for classification)
seq_dim = 28                    # Sequence length - How many inputs are viewed at once?
learning_rate = 0.1 

# instantiate model
model = LSTM(input_size, hidden_dim, layer_dim, output_size, True).to(device)
x = LSTM()
# Step 5: Instantiate Loss Class
criterion = nn.CrossEntropyLoss()

# Step 6: Instantiate Optimiser Class
optimiser = torch.optim.SGD(model.parameters(), learning_rate)


def parameter_analysis():
    print('Number of Model Parameters = ', len(list(model.parameters()))) 
    print(model.parameters().data)
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())
    '''
        model.parameters()[0] = input ---> gates: weights
        model.parameters()[1] = input ---> gates: bias
                          [2] = hidden state ---> gates: weights
                          [3] = hidden state ---> gates: bias
                          [4] = hidden state ---> output: weights
                          [5] = hidden state ---> output: bias
    '''

# Step 7: Train Model
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable
        # Changes dimension of tensor
        images = images.view(-1, seq_dim, input_size).requires_grad_().cuda()   # returns a new tensor of different shape.
                                                                                # -1 = any value that will fit with the other values.
                                                                                # seq_dim = row segmentation.
                                                                                # input_size = coloums (feature) segmentation. 
        labels = labels.cuda()

        # Clear gradients w.r.t. parameters
        optimiser.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimiser.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images as Variable
                images = images.view(-1, seq_dim, input_size).cuda()
                labels = labels.cuda()

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Epoch: {} | Iteration: {} | Loss: {} | Accuracy: {}'.format(epoch, iter, loss.item(), accuracy))
