#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np

path1 = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Practices\\ImageComparison\\peterfg.jpg'
path2 = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Practices\\ImageComparison\\peterfg2.jpg'
path3 = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Practices\\ImageComparison\\stewiefg.jpeg'

img1 = cv2.imread(path1)
img1 = cv2.resize(img1, (840, 640))

img2 = cv2.imread(path2)
img2 = cv2.resize(img2, (840, 640)) # size should be same if it's not it doesn't matter that the images match 

# if img1.shape == img2.shape:
#     print('Same size')
# if we dont know the sizes we can use this comparison first

diff = cv2.subtract(img1, img2)
"""
OpenCV checks or manipulates the images by pixel level because of this fact 
we can get the difference of the images in pixel level.
Before you subtract any image, you should note that the two images must be in the same size and depth. 
Otherwise, it will throw an error.
"""
b, g, r = cv2.split(diff)
# print(b)
# print(g)
# print(r)

# if cv2.countNonZero(b) == 0 & cv2.countNonZero(g) == 0 & cv2.countNonZero(r) == 0:
#     print('There is no difference.')
# else:
#     print('Images are not same.')

img3 = cv2.imread(path3)
img3 = cv2.resize(img3, (840, 640))
diff2 = cv2.subtract(img1, img3)



cv2.imshow('Difference between img1 and img2', diff) # it's just a black window because there is no difference
cv2.imshow('Difference between img1 and img3', diff2) 

cv2.waitKey(0)
cv2.destroyAllWindows()

