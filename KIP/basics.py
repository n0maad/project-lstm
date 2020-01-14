''' 
Tensor Basics
    
    Scalars: 3, 1.2
        are individual values.
        has a rank of zero.
        has no dimension.

    Vectors: [3, 4, 8]
        values represent a single element.
        has a rank of one as you need the one index to locate the element.
        has a single dimension of 3 (1, 3).

    Matrix: [[1, 2, 3],
             [4, 5, 6]] 
        values represent a single element?
        has a rank of two as you need two indexes to locate the single element.
        has a dimension of (2, 3) as there are two rows and three columns.

    Tensor: [ [[1, 2, 3],[4, 5, 6]], [[7, 8, 9],[10, 11, 12]] ] 
        values represent a single element.
        has a rank of three (or more) as you need three (or more) indexes to locate the single element.
        has a dimension of (2, 2, 3) as there are two matrices with two rows and three columns.
    
    Rank is simply the length of the dimension???
    dimension = (no. of matrices, no. of vectors(rows), no. of columns)??

'''

'''
    There are various Datatypes that have both a CPU and GPU tensor.
    E.g.
        32-bit floating point
            CPU: torch.FloatTensor - This is set as defulat
            GPU: torch.cuda.FloatTensor

        64-bit floating point
            CPU: torch.DoubleTensor
            GPU: torch.cuda.DubleTensor

        16-bit floating point
            CPU: torch.HalfTensor
            GPU: torch.cuda.HalfTensor
        
        8-bit integer(unsigned)
            CPU: torch.ByteTensor
            GPU: torch.cuda.ByteTensor
        
        8-bit integer(signed)
            CPU: torch.CharTensor
            GPU: torch.cuda.CharTensor

    The differece between the syntaxes is simply placing 'cuda' between the torch library name and datatype being used.
'''

import torch
import torch.nn as nn
import torch.cuda

def version():
    print('Pytorch version: ', torch.__version__)
    print('CUDA availability: ', torch.cuda.is_available())

def genTensor(z = 2, x = 3, y = 3):
    if z == 2: return torch.Tensor(x, y)    # creates a tensor consisting of two vectors with two columns
    if z == 1: return torch.ones(x,y)       # creates a tensor of ones.
    else: return torch.zeros(x, y)          # creates a tensor of zeroes.

def genTensor_like(x, y):
    if y == 1: return torch.ones_like(x)    # creates a tensor like x but replaced with ones
    if y == 0: return torch.zeros_like(x)   # creates a tensor like x but replaced with zeroes.
    else: return x                          # returns x

def tensorValues(t):
    print(t) 
    print('Tensor type: ', t.type())
    print('Tensor size: ', t.size())
    print('Tensor dimension: ', t.dim())    # returns rank of element(s)
    
def convertTensor(x, dtype):
    print(x)
    datatypes = {0:x.type(torch.IntTensor), 1:x.type(torch.DoubleTensor)}
    x = datatypes[dtype]
    print(x)
    return x

    
'''
    Tensor Quick Maths

    Inplace and out-of-place operations
        t = t1.add(t2) adds the tensors t1 and t2 together without altering the original tensors
        t = t1.add_(t2) adds the tensors t1 and t2 together and replaces t1 with the resultant

    You may also perform the operation on the tensor itself or envoke the function from the torch library
        t1.cos() or torch.cos(t1)

    torch.linspace(x,y, steps = z)
        creates a one dimensional tensor of steps equally spaced points(z) between the start (x) and end (y)
        torch.linspace(3, 10, steps = 5)
            3.0000
            4.7500
            6.5000
            8.25000
            10.0000 
    
'''






#main

