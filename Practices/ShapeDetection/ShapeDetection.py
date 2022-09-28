#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Practices\\ShapeDetection\\polygons.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
contours, ret = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # Second argument specify whether shape is a closed contour (if passed True), or just a curve.
    epsilon = 0.01*cv2.arcLength(cnt, True)
    # It approximates a contour shape to another shape with less number of vertices depending upon the precision we specify.
    # Second argument is called epsilon, which is maximum distance from contour to approximated contour.
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    cv2.drawContours(img, [approx], 0, (0), 5)
    
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    
    if len(approx) == 3:
        cv2.putText(img, 'triangle', (x, y), font, 0.5, (0))
    elif len(approx) == 4:
        cv2.putText(img, 'rectangle', (x, y), font, 0.5, (0))    
    elif len(approx) == 5:
        cv2.putText(img, 'pentagon', (x, y), font, 0.5, (0))
    else:
        cv2.putText(img, 'ellipse', (x, y), font, 0.5, (0))    


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

