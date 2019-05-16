#!/usr/bin/env python
# coding: utf-8

# In[15]:


import model
import numpy as np, pandas as pd,time, os,pickle,cv2,math,time,matplotlib.pyplot as plt,random
from PIL import Image
from PIL import ImageDraw
import torch
from torchsummary import summary
from skimage import io, transform


# In[28]:


files = [os.path.join(r,f) for r,d,files in os.walk("HandVeinDatabase") for f in files if 'bmp' in f]


# In[10]:


net = network()
p2cAll = {}
start = time.time()
for i,path in enumerate(files):
    image = io.imread(path)
    file = path.split("\\")[-1]
    p2cAll[file] = getPoint(image,net = net)

visualizeResult(15)


# In[11]:


with open('p2cAll.pickle', 'wb') as handle:
    pickle.dump(p2cAll, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[3]:


def network():
    net = model.Network()
    if torch.cuda.is_available():net.cuda()
    net.load_state_dict(torch.load('Loss 50.pth'))
    return net


# In[7]:


# Give an image, if with Image in will return the image with the points plotted
# Pass the network, or it will create the network
# Image Path of image array
def getPoint(img,withImage = False,net = None):
    if(type(img)==str):
        image = io.imread(img)
    else: image = img
    img = torch.tensor(image)
    img = img.transpose(1,2).transpose(0,1)
    x,y,z = img.shape
    if net == None:
        net = network()
    img = img.reshape((1,x,y,z)).type('torch.FloatTensor').cuda()
    net.eval()
    cor = net(img).detach().cpu().numpy()[0]
    cor = [(cor[0],cor[1]),(cor[2],cor[3])]
    if withImage:
        for x,y in cor:
            cv2.circle(image,(x,y),2,(255,0,0),-1)
        return cor, image
    else:
        return cor


# In[50]:


# Visualize n images with the key points generated 
def visualizeResult(n):
    global p2cAll,files
    imgs = random.sample(files, n)
    r = math.ceil(n/5)
    plt.figure(figsize=(30,15))
    for i,p in enumerate(imgs):
        plt.subplot(r,5,i+1)
        img = cv2.imread(p)
        for x,y in p2cAll[p.split("\\")[-1]]:
            cv2.circle(img,(x,y),3,(255,0,0),-1)
        plt.imshow(img)


# In[ ]:




