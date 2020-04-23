## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
                
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
#        self.pool = nn.MaxPool2d(2, 2)
#        self.fc1 = nn.Linear(128*28*28, 68*2)
#        self.pool = nn.MaxPool2d(5, 3)
#        self.fc1 = nn.Linear(128*13*13, 68*2)
#        self.fc1 = nn.Linear(128*13*13, 512)
#        self.fc2 = nn.Linear(512, 68*2)
#        self.fc1 = nn.Linear(256*6*6, 68*2)
        self.fc1 = nn.Linear(256*13*13, 256)
        #self.fc1 = nn.Linear(256*6*6, 256)
        self.fc2 = nn.Linear(256, 68*2)



        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        #x = self.pool(x)
        #x = self.pool(self.bn4(F.relu(self.conv4(x))))
        
        # a modified x, having gone through all the layers of your model, should be returned

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # linear layers
        #x = self.fc1(x)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        
        # limit range to [-1,1]
#        x = F.tanh(x)

        return x
