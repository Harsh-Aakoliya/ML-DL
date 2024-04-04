import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn #from this torch.nn we can create different layers

#finding best device on which we can run our tensor 
if torch.cuda.is_available():
    device=torch.device(type="cuda",index=0) #cuda is for GPU  and idx=0 means from multiple available GPUs assign first one i.e. at oth idx one
else:
  device=torch.device(type="cpu",index=0) #by defult it will run on CPU 


#we have two ways either we create tensor on GPU or create on CPU and later on we can move it to GPU for computation

print(device)

#all images are of 28x28
#60,000 for training 
training_data = datasets.MNIST(
    root = 'data', #dataset will be downloaded into "data" directory
    train = True,
    download = True, #if we run second time then it will not download again it will find it from "data" directory
    transform = ToTensor() # PIL images to numpy ndarrys, normalisation (0 to 1 ma lavshe), changes then shape like C x H x W (1 x 28 x 28)
)

#10,000 for testing 
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)

train_dataloader = DataLoader(
    training_data,
    batch_size = 64,
    shuffle = True
)

test_dataloader = DataLoader(
    test_data,
    batch_size = 64,
    shuffle = True
)

#this class inherits nn.module
class FNN(nn.Module):
  #we are creating various layers we are not joining those  hence we can write lines of init method in any manner 
  def __init__(self):
    super().__init__() #dendor metod 
    self.relu = nn.ReLU() #defining activation function and common for all layers
    # dence layer is also known as Linear layer
    #creating layer with 784 neurons and 512 layers and it will initialse the matrix for W=784*512 and B vector=512 will be initialisation 
    self.in_h1 = nn.Linear(in_features=784,out_features=512) #784=()
    self.b1 = nn.BatchNorm1d(num_features=512) #batchnormalisation is used for larger range of hyper-parameter (no of layers, no of epoch, leraring rate), and side-effect is it provides bit of side regurlisation and since we are using it at output so we have given 512 as no of feature
    self.h1_h2 = nn.Linear(in_features=512,out_features=256)
    self.b2  = nn.BatchNorm1d(num_features=256)
    self.h2_h3 = nn.Linear(in_features=256,out_features=128)
    self.b3 = nn.BatchNorm1d(num_features=128)
    self.h3_h4 = nn.Linear(in_features=128,out_features=64)
    self.b4 = nn.BatchNorm1d(num_features=64)
    self.h4_h5 = nn.Linear(in_features=64,out_features=32)
    self.b5 = nn.BatchNorm1d(num_features=32)
    self.h5_h6 = nn.Linear(in_features=32,out_features=10)
    self.b6 = nn.BatchNorm1d(num_features=10)
    return


  #Now here order is must because it will create NN in that order 
  def forward(self,x): # x is 64 x (784) and in_h1 will perform for all 64 images one by one 
    x = self.in_h1(x) #now x will become 64 X (512)
    x = self.b1(x) #and we are applying normalisation to output 64 X (512) and BN is applied (generaly) before relu 
    x = self.relu(x) 
    x = self.h1_h2(x)
    x = self.b2(x)
    x = self.relu(x)
    x = self.h2_h3(x)
    x = self.b3(x)
    x = self.relu(x)
    x = self.h3_h4(x)
    x = self.b4(x)
    x = self.relu(x)
    x = self.h4_h5(x)
    x = self.b5(x)
    x = self.relu(x)
    x = self.h5_h6(x)
    x = self.b6(x) #so here we have 64 X 10 matrix for x and we are not applying activation function instead of that we are using softmax 
    return x
  




#method for trining for one epoch 
def train_fn(model,loss,optim,train): #model is instence of NN, loss function like after one epoch we will calculate loss here we are using categorical cross entropy, optim is for optimasation(or which type of GD) like SGD, ADAM etc.. , train is which dataloader we are using
  model.train()#initialize model for training 
  final_acc=0 #for one epoch 
  #if we don't do enumerate then it will return (image and label) but if we do then  it it will return batch no also 
  for i,(data,label) in enumerate(train): #here data is image of (28 x 28)
    data=data.reshape(-1,28*28) # we are converting it into 1D array of 28*28, -1 for infer dimentionality like we have 64*1*28*28 = y * (1*28*28)  so infered dimentionality will be 64 so img will be 64 x 784 

    #sending images and label on available devices 
    data=data.to(device)
    label=label.to(device)


    optim.zero_grad()#it will prevent the accumulation of gradient since here we are dealing with batches To avoid this accumulation and ensure that each call to backward() 
#         computes gradients only for the current batch or computation step, you need to reset the gradients to zero. 
#         That's precisely the purpose of optimizer.zero_grad(). It clears the gradients of 
#         all optimized tensors so that you can compute the gradients for the new batch or computation step.
    pred=model(data) #pred will be having 64 x 10 means for 64 images what is probability of each 10 classes  
    loss=loss_fn(pred,label)
    loss.backward()
    optim.step()
    pred=torch.argmax(pred,dim=1) #finding the index of maximum probability so after that we will have 64 values in pred  
    acc=(pred==label).sum().item() 
    final_acc+=acc    
    
  print("Epoch Accuracy :",final_acc/(16*len(train)))


def eval_fn(model,loss,test):
  model.eval()
  final_acc=0
  with torch.no_grad():
    for i,(data,label) in enumerate(test):
      data=data.reshape(-1,28*28) # we are converting it into 1D array of 28*28, -1 for infer dimentionality like we have 64*1*28*28 = y * (1*28*28)  so infered dimentionality will be 64 so img will be 64 x 784 
      data=data.to(device)
      label=label.to(device)
      pred=model(data)
      loss=loss_fn(pred,label)
      pred=torch.argmax(pred,dim=1)
      acc=(pred==label).sum().item()
      final_acc+=acc
    print("Testing Accuracy :",final_acc/(16*len(test)))

model=FNN().to(device)
loss_fn=nn.CrossEntropyLoss()
optim=torch.optim.Adam(model.parameters(),lr=0.001)
epochs=10
for epoch in range(epochs):
  train_fn(model,loss_fn,optim,train_dataloader)
eval_fn(model,loss_fn,test_dataloader)


