#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[3]:


# pixels
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\scooby-doo.jpg')

dimension = img.shape
print(dimension)

color = img[150, 200] # 150-200 th pixel's bgr codes -> 83,69,47
print(color)

cv2.imshow('scooby-doo', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


# roi -> region of interest
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\scooby-doo.jpg')

roi = img[10:250, 400:700]

cv2.imshow('scooby-doo', img)
cv2.imshow('ROI', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


# sum of images

circle = np.zeros((512, 512, 3), np.uint8) + 255
cv2.circle(circle, (256, 256), 60, (0, 0, 255), -1) # circle

rectangle = np.zeros((512, 512, 3), np.uint8) + 255
cv2.rectangle(rectangle, (150, 150), (350, 350), (255, 0, 0), -1) # rectangle

add = cv2.add(circle, rectangle)

cv2.imshow('Circle', circle)
cv2.imshow('Rectangle', rectangle)
cv2.imshow('Sum', add)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


# weighted sum of images -> f(x, y) = x*a +y*b +c

circle = np.zeros((512, 512, 3), np.uint8) + 255
cv2.circle(circle, (256, 256), 60, (0, 0, 255), -1) # circle

rectangle = np.zeros((512, 512, 3), np.uint8) + 255
cv2.rectangle(rectangle, (150, 150), (350, 350), (255, 0, 0), -1) # rectangle

dst = cv2.addWeighted(circle, 0.7, rectangle, 0.3, 0)

cv2.imshow('Circle', circle)
cv2.imshow('Rectangle', rectangle)
cv2.imshow('Dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


# color spaces
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\scooby-doo.jpg')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('scooby-doo', img)
cv2.imshow('scooby-doo RGB', img1)
cv2.imshow('scooby-doo LUV', img2)
cv2.imshow('scooby-doo HSV', img3)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


# changing color spaces of videos
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flip_frame = cv2.flip(frame,1)
    cv2.imshow('webcam', flip_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[5]:


# HSV finder (HSV trackbar)
cap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow('Trackbar')
cv2.resizeWindow('Trackbar', 400, 800)

cv2.createTrackbar('Lower-H', 'Trackbar', 0, 180, nothing)
cv2.createTrackbar('Lower-S', 'Trackbar', 0, 255, nothing)
cv2.createTrackbar('Lower-V', 'Trackbar', 0, 255, nothing)

cv2.createTrackbar('Upper-H', 'Trackbar', 0, 180, nothing)
cv2.createTrackbar('Upper-S', 'Trackbar', 0, 255, nothing)
cv2.createTrackbar('Upper-V', 'Trackbar', 0, 255, nothing)

cv2.setTrackbarPos('Upper-H', 'Trackbar', 180) # to start Upper H slide from 180
cv2.setTrackbarPos('Upper-S', 'Trackbar', 255)
cv2.setTrackbarPos('Upper-V', 'Trackbar', 255)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_h = cv2.getTrackbarPos('Lower-H', 'Trackbar')
    lower_s = cv2.getTrackbarPos('Lower-S', 'Trackbar')
    lower_v = cv2.getTrackbarPos('Lower-V', 'Trackbar')
    
    upper_h = cv2.getTrackbarPos('Upper-H', 'Trackbar')
    upper_s = cv2.getTrackbarPos('Upper-S', 'Trackbar')
    upper_v = cv2.getTrackbarPos('Upper-V', 'Trackbar')
    
    lower_color = np.array([lower_h, lower_s, lower_v])
    upper_color = np.array([upper_h, upper_s, upper_v])
    
    mask = cv2.inRange(frame_hsv, lower_color, upper_color)
    cv2.imshow('original', frame)
    cv2.imshow('masked', mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[5]:


# bitwise operators
import cv2 

img1 = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\bitwise_1.png')
img2 = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\bitwise_2.png')

bit_and = cv2.bitwise_and(img2, img1)
bit_or = cv2.bitwise_or(img2, img1)
bit_not2 = cv2.bitwise_not(img2)

cv2.imshow('Original img1 ', img1)
cv2.imshow('Original img2 ', img2)

cv2.imshow('And', bit_and)
cv2.imshow('Or', bit_or)
cv2.imshow('Not', bit_not2)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[19]:


# matrix
import numpy as np

img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\scooby-doo.jpg', 0) # 0 stands for GRAYSCALE
row, col = img.shape # row -> 545 col -> 1200

M = np.float32([[1, 0, 10], [0, 1, 70]])
# M = np.float32([[1, 0, 50], [0, 1, 70]])
# M = np.float32([[1, 0, 10], [0, 1, 20]])
# M = np.float32([[1, 0, 50], [0, 1, 70]])

dst = cv2.warpAffine(img, M, (row, col))

cv2.imshow('dst', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[22]:


# rotation
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\scooby-doo.jpg', 0) 
row, col = img.shape # row -> 545 col -> 1200

# cv2.getRotationMatrix2D(center, angle, scale)
M = cv2.getRotationMatrix2D((col/5, row/2), 90, 0.5)

dst = cv2.warpAffine(img, M, (col, row))

cv2.imshow('dst', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[32]:


# thresholding
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\scooby-doo.jpg', 0) 

ret , th1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
"""
The method returns two outputs. The first is the threshold that was used and the second output is the thresholded image.
cv2.threshold(src, thresholdValue, maxVal, thresholdingTechnique)
"""
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY ,11, 2)
"""
Adaptive thresholding is the method where the threshold value is calculated for smaller regions. 
This leads to different threshold values for different regions with respect to the change in lighting.
cv2.adaptiveThreshold(source, maxVal, adaptiveMethod, thresholdType, blocksize, constant)
"""
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY ,11, 2)

cv2.imshow('img', img)
cv2.imshow('thresholded img1 ', th1)
cv2.imshow('thresholded img2 ', th2)
cv2.imshow('thresholded img3 ', th3)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[44]:


# erosion - dilation - opening
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\scooby-doo.jpg', 0) 

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=3)

dilation = cv2.dilate(img, kernel, iterations=5)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN,kernel)
"""
opening is just another name of erosion followed by dilation. It is useful in removing noise.
"""
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT,kernel)

cv2.imshow('img', img)
cv2.imshow('erosion ', erosion)
cv2.imshow('dilation ', dilation)
cv2.imshow('opening', opening)
cv2.imshow('gradient', gradient)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[57]:


# histogram
import matplotlib.pyplot as plt

# img = np.zeros((500, 500), np.uint8)
# cv2.rectangle(img, (50, 80), (150, 450), (255, 150, 30), -1)
# cv2.circle(img, (250, 250), 80, (85, 150, 55), -3)
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\scooby-doo.jpg') 
b, g, r = cv2.split(img)


cv2.imshow('img', img)

# .ravel() -> A 1-D array, containing the elements of the input, is returned.
# plt.hist(img.ravel(), 256, [0, 256])
plt.hist(b.ravel(), 256, [0, 256], label='blue')
plt.hist(g.ravel(), 256, [0, 256], label='green')
plt.hist(r.ravel(), 256, [0, 256], label='red')
plt.legend()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[67]:


# shi-tomasi corner detection
img = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\text.png')
img1 = cv2.imread('C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\OpenCVOperations\\contour.png')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
# corners = np.int0(corners)

# for corner in corners:
#     x, y = corner.ravel()
#     cv2.circle(img, (x, y), 3, (0,255,50), -1)

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img1, (x, y), 3, (0,255,50), -1)
    
cv2.imshow('img', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[70]:


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read() 
    frame = cv2.flip(frame, 1)
    # cv2.canny(input, minthreshold, maxthreshold)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame, (x, y), 3, (0,255,50), -1)
        
    cv2. imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:


# canny edge detection

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read() 
    frame = cv2.flip(frame, 1)
    # cv2.canny(input, minthreshold, maxthreshold)
    edges = cv2.Canny(frame, 100, 200)
    cv2. imshow('Frame', frame)
    cv2.imshow('Edges', edges)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

