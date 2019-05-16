#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np, pandas as pd,time, os,pickle,cv2,math,os,matplotlib.pyplot as plt,random
from PIL import Image,ImageDraw


# In[133]:


files = [os.path.join(r,f) for r,d,files in os.walk("HandVeinDatabase") for f in files if 'bmp' in f]


# In[136]:


class roi:
    def __init__(self):
        #file to co ordinates
        self.p2c = {}
        if os.path.isfile('p2cAll.pickle'):
            with open('p2cAll.pickle', 'rb') as f:
                self.p2c = pickle.load(f)
        else:
            print("No hand Measurement data found")
    
    # given the path to an image it will rotate and give bounding box to roi
    def getBoundingBox(self,path):
        p = path.split('\\')[-1]
        cor = self.p2c[p]  # old pointf of mid fingers
        img = Image.open(path)
        d = self.angel(cor) # angel of to points
        cor = self.new_points(cor,d,img.size)
        x1,y1 = cor[0]
        x2,y2 = cor[1]
        dis = math.sqrt((x1-x2)**2 + (y1-y2)**2) #length of mid finger 
        img = img.rotate(d)
        draw = ImageDraw.Draw(img)
        for x,y in cor:
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(255,0,0,255))
        by1 = y1+(y1-y2)*.08
        by2 = y1+y1-y2
        if "left" in p:
            bx1 = x1 - (dis//3)*1.7
            bx2 = x1 + (dis//3)*.9
        else:
            bx1 = x1 - (dis//3)*.8
            bx2 = x1 + (dis//3)*1.8
        bb = [(bx1,by1),(bx2,by2)]
        draw.rectangle(bb,outline="red")
#         cv2.rectangle(img, (bx1, by1), (bx2, by2), (255,0,0), 2)
#         plt.imshow(np.array(img))
        return np.array(img)
    def new_points(self,cor,d,imsize):
        xm, ym = imsize
        a = -(d*math.pi)/180
        xm = xm//2
        ym = ym//2 
        new_cor = []
        for (x,y) in cor:
            xr = (x - xm) * math.cos(a) - (y - ym) * math.sin(a)   + xm
            yr = (x - xm) * math.sin(a) + (y - ym) * math.cos(a)   + ym
            new_cor.append((xr,yr))
        return new_cor
    def angel(self,cor):
        x1,y1 = cor[0]
        x2,y2 = cor[1]
        m = -(y1-y2)/(x1-x2)
        d = 90-math.atan(m)*(180/math.pi)
        if d > 90:
            d = -(180-d)
        return d
    # Visualize n images with the key points generated 
    def visualizeResult(self,n):
        global files
        imgs = random.sample(files, n)
        r = math.ceil(n/5)
        plt.figure(figsize=(30,15))
        
        for i,p in enumerate(imgs):
            plt.subplot(r,5,i+1)
            img = self.getBoundingBox(imgs[i])
            plt.imshow(img)


# In[ ]:


roi = roi()


# In[139]:


roi.visualizeResult(15)


# In[137]:





# In[ ]:




