#!/usr/bin/env python
# coding: utf-8

# In[2]:


# HOUGH LINE TRANSFORM
import cv2
import numpy as np

img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\HoughTransforms\\h_line.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# canny detects the edges
edges = cv2.Canny(gray, 75, 150)
"""
Syntax: cv2.Canny(image, T_lower, T_upper, aperture_size, L2Gradient)

Image: Input image to which Canny filter will be applied
T_lower: Lower threshold value in Hysteresis Thresholding
T_upper: Upper threshold value in Hysteresis Thresholding
aperture_size: Aperture size of the Sobel filter.
L2Gradient: Boolean parameter used for more precision in calculating Edge Gradient.
"""
# hough lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=150)
"""
 It simply returns an array of (r, 0) values. r is measured in pixels and 0 is measured in radians. 
"""
# print(lines)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


# video
"""
related mp4 file is to large so a problem occurs when pushing to the repo
"""

# cap = cv2.VideoCapture('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\HoughTransforms\\line.mp4')

# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (640, 480))
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     # hsv range for yellow
#     lower_yellow = np.array([18, 94, 140], np.uint8) 
#     upper_yellow = np.array([48, 255, 255], np.uint8)
    
#     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
#     edges = cv2.Canny(mask, 75, 250)
#     # cv2.imshow('mask', mask)
#     # cv2.imshow('edges', edges)
    
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=25)
    
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
#     cv2.imshow('video', frame)    
#     if cv2.waitKey(3) & 0xFF == ord('q'):
#         break
        
# cap.release()
# cv2.destroyAllWindows()

