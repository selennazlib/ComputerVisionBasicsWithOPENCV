#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('settings')

cv2.createTrackbar('Lower-hue', 'settings', 0, 180, nothing)
cv2.createTrackbar('Lower-saturation', 'settings', 0, 255, nothing)
cv2.createTrackbar('Lower-value', 'settings', 0, 255, nothing)
cv2.createTrackbar('Upper-hue', 'settings', 0, 180, nothing)
cv2.createTrackbar('Upper-saturation', 'settings', 0, 255, nothing)
cv2.createTrackbar('Upper-value', 'settings', 0, 255, nothing)

while True:
    ret, frame = cap.read()
    frame =cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lh = cv2.getTrackbarPos('Lower-hue', 'settings')
    ls = cv2.getTrackbarPos('Lower-saturation', 'settings')
    lv = cv2.getTrackbarPos('Lower-value', 'settings')
    uh = cv2.getTrackbarPos('Upper-hue', 'settings')
    us = cv2.getTrackbarPos('Upper-saturation', 'settings')
    uv = cv2.getTrackbarPos('Upper-value', 'settings')
    
    
    lower_color = np.array([lh, ls, lv])    
    upper_color = np.array([uh, us, uv])
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    
    contours, ret = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        epsilon = 0.02*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        
        if area > 400:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
            if len(approx) ==3:
                cv2.putText(frame, 'triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            elif len(approx) ==4:
                cv2.putText(frame, 'rectangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            elif len(approx) ==5:
                cv2.putText(frame, 'pentagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            else:
                cv2.putText(frame, 'circle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                
    cv2.imshow('frame',frame)         
    cv2.imshow('mask', mask)
    
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

