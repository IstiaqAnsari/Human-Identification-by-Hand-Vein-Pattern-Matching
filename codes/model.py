import torch.nn.functional as F
import torch, torch.nn as nn, torch.optim as optim


class Network(nn.Module):
    def __init__(self,d = False):
        self.d = d
        super(Network,self).__init__()
        
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout2d(.2)
        
        self.conv1 = nn.Conv2d(3,64,kernel_size = 9,stride =2 ,padding = 1)
        self.conv11 = nn.Conv2d(64,64,kernel_size=3,stride = 1,padding = 1)
        self.norm1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv2 = nn.Conv2d(64,64,kernel_size = 2,stride =2 ,padding = 1)
        self.conv21 = nn.Conv2d(64,64,kernel_size = 3,stride =1 ,padding = 1)
        self.conv22 = nn.Conv2d(64,64,kernel_size = 3,stride =1 ,padding = 1) 
        self.norm2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size = 2,stride =2 ,padding = 1)
        self.conv31 = nn.Conv2d(128,128,kernel_size = 3,stride =1 ,padding = 1)
        self.conv32 = nn.Conv2d(128,128,kernel_size = 3,stride =1 ,padding = 1) 
        self.norm3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv4 = nn.Conv2d(128,256,kernel_size = 2,stride =2 ,padding = 1)
        self.conv41 = nn.Conv2d(256,256,kernel_size = 3,stride =1 ,padding = 1)
        self.conv42 = nn.Conv2d(256,256,kernel_size = 3,stride =1 ,padding = 1) 
        self.norm4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv5 = nn.Conv2d(256,256,kernel_size = 2,stride =2 ,padding = 1)
        self.conv51 = nn.Conv2d(256,256,kernel_size = 3,stride =1 ,padding = 1)
        self.conv52 = nn.Conv2d(256,256,kernel_size = 3,stride =1 ,padding = 1) 
        self.norm5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.conv6 = nn.Conv2d(256,256,kernel_size = 2,stride =2 ,padding = 1)
        self.conv61 = nn.Conv2d(256,256,kernel_size = 3,stride =1 ,padding = 1)
        self.conv62 = nn.Conv2d(256,256,kernel_size = 3,stride =1 ,padding = 1) 
        self.norm6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
#         self.poolx = nn.MaxPool2d(kernel_size=(5,1),stride = 1,padding = 0)
#         self.pooly = nn.MaxPool2d(kernel_size=(1,6),stride = 1,padding = 0)
        
#         self.fc1 = nn.Linear(1536,256)
#         self.fc2 = nn.Linear(1280,256)
        
        self.fc00 = nn.Linear(7680,512)
        self.fc11 = nn.Linear(512,32)
        self.fc22 = nn.Linear(32,4)
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv11(x))
        x = self.norm1(x)
        if self.d:x = self.drop(x)
        
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv21(x))
        x = self.relu(self.conv22(x))
        x = self.norm2(x)
        if self.d:x = self.drop(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv31(x))
        x = self.relu(self.conv32(x))
        x = self.norm3(x)
        if self.d:x = self.drop(x)
        
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv41(x))
        x = self.relu(self.conv42(x))
        x = self.norm4(x)
        if self.d:x = self.drop(x)
        
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv51(x))
        x = self.relu(self.conv52(x))
        x = self.norm5(x)
        if self.d:x = self.drop(x)
        
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv61(x))
        x = self.relu(self.conv62(x))
        x = self.norm6(x)
        if self.d:x = self.drop(x)
        
#         h = self.poolx(x)
#         v = self.pooly(x)
#         h = h.view(h.size(0), -1)
#         v = v.view(v.size(0), -1)
        
        
#         h = self.relu(self.fc1(h))
#         v = self.relu(self.fc2(v))
#         x = torch.cat((h,v),1)
        x = x.view(x.size(0),-1)
        x = self.relu(self.fc00(x))
        x = self.relu(self.fc11(x))
        x = self.fc22(x)
        return x