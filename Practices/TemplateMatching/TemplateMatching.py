#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import numpy as np

path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Practices\\TemplateMatching\\stewie.jpg'
template_path = 'C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Practices\\TemplateMatching\\stewie1.png'

img = cv2.imread(path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread(template_path, 0)
w, h = template.shape[::-1]

result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
# print(result)
loc = np.where(result >= 0.9)
# print(loc)

for point in zip(*loc[::-1]):
    # print(point)
    cv2.rectangle(img, point, (point[0]+w, point[1]+h), (255, 255, 0), 3)
    
    
    


cv2.imshow('Original Image', img)
# cv2.imshow('Image', template)
# cv2.imshow('Result', result)


cv2.waitKey(0)
cv2.destroyAllWindows()

