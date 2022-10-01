#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import numpy as np

def nothing(x):
    pass

path1 = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Practices\\ImageTransition\\bunny.jfif'
path2 = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Practices\\ImageTransition\\bbunny.jfif'

img1 = cv2.imread(path1)
img1 = cv2.resize(img1, (640, 640))

img2 = cv2.imread(path2)
img2 = cv2.resize(img2, (640, 640))

output = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
windowName = 'Transition'
cv2.namedWindow(windowName)

cv2.createTrackbar('Alpha Beta', windowName, 0, 1000, nothing)

while True:
    cv2.imshow(windowName, output)
    
    alpha = cv2.getTrackbarPos('Alpha Beta', windowName) / 1000
    beta = 1 - alpha
    output = cv2.addWeighted(img1, alpha, img2, beta, 0)
    
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        


cv2.waitKey(0)
cv2.destroyAllWindows()

