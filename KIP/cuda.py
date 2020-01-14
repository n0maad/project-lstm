import torch 

# Check to see if a GPU is available.
def cudaInfo():
    if torch.cuda.is_available():
        x = torch.cuda.device_count()
        print('This compter has', x, 'GPU available.')
        for i in range(x):     
            print(torch.cuda.get_device_name(i))
    else: print('No GPU is available')

# moving a tensor to the GPU fron CPU
t = torch.Tensor([2,3])     # tensor is stored on the CPU
print('CPU: ',t)
t = t.cuda(0)               # tensor is moved to GPU @ index 0.
print('GPU: ',t)               # indicates the tensor is on the GPU

# bring it back to CPU
t = t.cpu()
print('CPU: ',t)

# creating a tensor directly on the GPU
t = torch.cuda.FloatTensor([2, 4])      # when generating a tensor on the GPU a datatype must be stated.
print('GPU: ',t)                               

# Allocating GPU
with torch.cuda.device(0):              # by wrapping the tensor generation within this statement all tensors will be stored on the indexed GPU
    t = torch.cuda.FloatTensor([2, 4])
    print(t)

    