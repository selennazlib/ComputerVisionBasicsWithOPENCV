#!/usr/bin/env python
# coding: utf-8

# In[27]:


# contours
"""
Contours can be explained simply as a curve joining all the continuous points (along the boundary), 
having same color or intensity. 
"""
import cv2

img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Contour-ConvexHull-ConvexityDefects\\contour1.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, ret = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)

cv2.drawContours(img ,contours, -1 , (0, 0, 255), 3)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[28]:


# object tracking
import numpy as np
cap = cv2.VideoCapture('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Contour-ConvexHull-ConvexityDefects\\dog.mp4')

while True:
    
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # hsv range for white
    sensitivity = 15
    lower_white= np.array([0, 0, 255 - sensitivity])
    upper_white= np.array([255, sensitivity, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()


# In[29]:


# image moments
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Contour-ConvexHull-ConvexityDefects\\contour.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

M = cv2.moments(thresh)
# print(M)
"""

{'m00': 15946170.0, 'm10': 2512158510.0, 'm01': 2140943280.0, 'm20': 568586636580.0, 
'm11': 337099003380.0, 'm02': 422375441400.0, 'm30': 144203744893170.0, 'm21': 76698286353210.0, 'm12': 66530197502520.0, 
'm03': 95616562341210.0, 'mu20': 172821360067.49454, 'mu11': -185051680.20017385, 'mu02': 134930987454.18369, 
'mu30': 176115778254.94254, 'mu21': 417776338746.10736, 'mu12': 38890234024.70071, 'mu03': 2676311195639.5464, 
'nu20': 0.0006796489325593326, 'nu11': -7.277467141054799e-07, 'nu02': 0.0005306386985763649, 
'nu30': 1.7344320571703827e-07, 'nu21': 4.1143654579295836e-07, 'nu12': 3.8300071277915876e-08, 'nu03': 2.635697936139322e-06}
"""
X = int(M['m10'] / M['m00'])
Y = int(M['m01'] / M['m00'])

cv2.circle(img, (X, Y) , 5, (25, 250, 255), -1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[30]:


# contour area
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Contour-ConvexHull-ConvexityDefects\\contour.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, ret = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

area = cv2.contourArea(cnt)
print(area)
M = cv2.moments(cnt)
print(M['m00'])

perimeter = cv2.arcLength(cnt,True)
print(perimeter)

cv2.imshow('original', img)
cv2.imshow('gray', gray)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[31]:


# convex hull
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Contour-ConvexHull-ConvexityDefects\\map.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray, (3, 3))

ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

contours, ret = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull = []

for i in range(len(contours)):
    hull.append(cv2.convexHull(contours[i], False))
    
bg = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

for i in range(len(contours)):
    cv2.drawContours(bg, contours, i, (255, 0, 0), 3, 8)
    cv2.drawContours(bg, hull, i, (255, 155, 0), 2, 8)



cv2.imshow('bg', bg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[41]:


# convexity defects
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\Contour-ConvexHull-ConvexityDefects\\star.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 0)
contours, ret = cv2.findContours(thresh, 2, 1)

cnt = contours[0]
hull = cv2.convexHull(cnt, returnPoints=False)

defect = cv2.convexityDefects(cnt, hull)

for i in range(defect.shape[0]):
    s, e, f, d = defect[i, 0] # start point- end point - furthest point - distance
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    
    cv2.line(img, start, end, [0, 255, 0], 2)
    cv2.circle(img, far, 5, [0, 255, 0], -1)
    
    
    
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()    

