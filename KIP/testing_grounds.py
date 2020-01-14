import pandas as pd
import torch
import random
import numpy as np


# Creating a custom dataset
datafile = pd.read_csv('../Data/Iris/iris_data.csv')        # import data as pd.dataframe.
datafile = datafile.sample(frac=1).reset_index(drop=True)   # shuffle data and reset the index values.

# Divide data intro training(80%) and testing(20%).
train = datafile.loc[0:len(datafile)*0.8-1, :]           
test  = datafile.loc[len(datafile)*0.8:len(datafile), :]

train_labels = train.iloc[:,4]                           # creates a df of just labels.
train_values = train.iloc[:,0:4]                         # drops labels from datafile.

train_values = torch.Tensor(train_values.values)
train_labels = torch.LongTensor(train_labels.values)

print(train_labels.shape)
'''
# Divide data intro training(80%) and testing(20%).
train = datafile.loc[0:len(datafile)*0.8-1, :]           # creates df of training values.
test = datafile.loc[len(datafile)*0.8:len(datafile), :]  # creates df of training labels.

# Seperate df into values and labels
train_labels = train.iloc[:,4]                                          # creates a df of just labels.
train_values = train.drop("species", axis=1, inplace = True)            # drops labels from train_values.

test_labels = test.iloc[:,4]
test_values = test.drop("species", axis=1, inplace = True) 
# creating tensor from targets_df 
#train_ds = torch.tensor(train.values)
#test_ds = torch.tensor(test.values)
'''
# printing out result
#print(len(test_ds))
'''
# Divide data intro training(80%) and testing(20%).
train = datafile.loc[0:len(datafile)*0.8-1, :]           
test = datafile.loc[len(datafile)*0.8:len(datafile), :]

# Convert dataframe into np.array 
train_values = np.array(train.iloc[:,0:4])                  # creates array for training values
train_labels = np.array(train.iloc[:,4])                    # creates array for training labels
train_labels = np.split(train_labels, len(train_labels))    # divides each element into its own array
train_labels = np.array(train_labels)                       # converts back to np.array as previous line converted values into a list.

test_values = np.array(test.iloc[:,0:4])                  # creates array for testing values
test_labels = np.array(test.iloc[:,4])                    # creates array for testing labels
test_labels = np.split(test_labels, len(test_labels))     # divides each element into its own array
test_labels = np.array(test_labels)                       # converts back to np.array as previous line converted values into a list.

# fill out labels to match values shape
fill = np.zeros([len(train_labels), train_values.shape[1]-1 ])   # creates an array of zeros 
train_labels = np.c_[train_labels, fill]                         # creates new array to match dimension of train_values

fill = np.zeros([len(test_labels), test_values.shape[1]-1 ])    # creates an array of zeros 
test_labels = np.c_[test_labels, fill]                          # creates new array to match dimension of test_values

# Convert data into tensors
train_ds = torch.tensor(np.array([[train_values], [train_labels]]))
test_ds = torch.tensor(np.array([[test_values], [test_labels]]))
'''
