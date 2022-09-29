#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np

img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Practices\\BackgroundSubtraction\\sponge.jpg')
img = cv2.resize(img, (640, 640))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

diff = cv2.absdiff(gray, blur)

ret, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

cv2.imshow('img', img)
cv2.imshow('diff', diff)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[15]:


img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Practices\\BackgroundSubtraction\\sponge.jpg', cv2.IMREAD_GRAYSCALE)
subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    mask = subtractor.apply(img)

    cv2.imshow('img', img)
    cv2.imshow('diff', mask)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

