#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2 

path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Practices\\BluryImageDetection\\stewiefg.jpeg'
img = cv2.imread(path)

blurry_img = cv2.medianBlur(img, 7)

cv2.imshow('Image', img)
cv2.imshow('Blurred Image', blurry_img)

laplacian = cv2.Laplacian(blurry_img, cv2.CV_64F).var()
# print(laplacian)
if laplacian < 500: # 500 is a threshold value that we chose
    print('Blurry image')

cv2.waitKey(0)
cv2.destroyAllWindows()

