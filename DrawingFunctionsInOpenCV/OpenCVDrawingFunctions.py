#!/usr/bin/env python
# coding: utf-8

# In[26]:


# creating canvas
import numpy as np

canvas = np.zeros((512, 512, 3), dtype=np.uint8)
print(canvas[:5])


# In[27]:


import cv2 
# black canvas
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[28]:


# white canvas
canvas = np.zeros((512, 512, 3), dtype=np.uint8) + 255
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[29]:


print(canvas)


# In[30]:


# cv2 drwaing functions
# line 
cv2.line(canvas, (0, 0), (512, 512), (216, 191, 216), thickness=3) # canvas->start point->end point->color->thickness
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[31]:


# rectangle
cv2.rectangle(canvas, (20, 20), (50, 50), (216, 191, 216), thickness=-1) # to fill the rectangle I set the thickness -1
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[32]:


# circle
cv2.circle(canvas, (250, 250), 100, (216, 191, 216), thickness=1) # canvas->center->radius->color->thickness
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[33]:


# triangle
p1 = (100, 200)
p2 = (50, 50)
p3 = (300, 100)
cv2.line(canvas, p1, p2, (216, 191, 216), thickness=1) 
cv2.line(canvas, p2, p3, (216, 191, 216), thickness=1)
cv2.line(canvas, p3, p1, (216, 191, 216), thickness=1) 
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[35]:


# text 
canvas = np.zeros((512, 512, 3), dtype=np.uint8) + 255

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

cv2.putText(canvas, "OpenCV", (30, 100), font, 3, (0, 0, 0), cv2.LINE_AA) # canvas->coordinates->font->font size->color->type of the text

cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[46]:


# trackbar
def nothing(x):
    pass

canvas = np.zeros((512, 512, 3), dtype=np.uint8) 
cv2.namedWindow('Canvas')

cv2.createTrackbar('R', 'Canvas', 0, 255, nothing) # R helps to slide the R value from 0 to 255
cv2.createTrackbar('G', 'Canvas', 0, 255, nothing)
cv2.createTrackbar('B', 'Canvas', 0, 255, nothing)

switch = '0: OFF, 1: ON'
cv2.createTrackbar(switch, 'Canvas', 0, 1, nothing)

while True:
    cv2.imshow('Canvas', canvas)
    if  cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    r = cv2.getTrackbarPos('R','Canvas')
    g = cv2.getTrackbarPos('G','Canvas')
    b = cv2.getTrackbarPos('B','Canvas')
    s = cv2.getTrackbarPos(switch,'Canvas')
    
    if s == 0:
        canvas[:] = [0, 0, 0]
    if s == 1:
        canvas[:] = [b, g, r]


cv2.destroyAllWindows()


# In[ ]:




