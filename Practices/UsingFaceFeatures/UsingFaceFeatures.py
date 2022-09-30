#!/usr/bin/env python
# coding: utf-8

# In[33]:


# FACE FEATURES
import cv2
import numpy as np


cv2.namedWindow('trackbar')
def nothing(x):
    pass

cv2.createTrackbar('LH', 'trackbar', 0, 179, nothing)
cv2.createTrackbar('LS', 'trackbar', 0, 255, nothing)
cv2.createTrackbar('LV', 'trackbar', 0, 255, nothing)
cv2.createTrackbar('UH', 'trackbar', 0, 179, nothing)
cv2.createTrackbar('US', 'trackbar', 0, 255, nothing)
cv2.createTrackbar('UV', 'trackbar', 0, 255, nothing)


def findMaxContour(contours):
    max_i = 0
    max_area = 0
    for i in range(len(contours)):
        area_face = cv2.contourArea(contours[i])
        
        if max_area < area_face:
            max_area = area_face
            max_i = i
            
        try:
            
            c = contours[max_i]
            
        except:
            
            contours = [0]
            c = contours[0]
            
        return c

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    roi = frame[50:250, 150:350] # frame [y1:y2, x1:,x2]
    cv2.rectangle(frame, (150, 50), (350, 250), (255, 0, 0), 0)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lh = cv2.getTrackbarPos('LH', 'trackbar')
    ls = cv2.getTrackbarPos('LS', 'trackbar')
    lv = cv2.getTrackbarPos('LV', 'trackbar')
    uh = cv2.getTrackbarPos('UH', 'trackbar')
    us = cv2.getTrackbarPos('US', 'trackbar')
    uv = cv2.getTrackbarPos('UV', 'trackbar')
    
    lower_color = np.array([lh, ls, lv])
    upper_color = np.array([uh, us, uv])
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 15)
    
    contours, ret = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        c = findMaxContour(contours)
            
        extLeft = tuple(c[c[:,:,0].argmin()][0])
        extRight = tuple(c[c[:,:,0].argmax()][0])
        extTop = tuple(c[c[:,:,1].argmin()][0])

        cv2.circle(roi, extLeft, 5, (0, 255, 0), 2)
        cv2.circle(roi, extRight, 5, (0, 255, 0), 2)
        cv2.circle(roi, extTop, 5, (0, 255, 0), 2)


    
    cv2.imshow('Frame', frame)
    cv2.imshow('Roi', roi)
    cv2.imshow('Mask', mask)
    
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

