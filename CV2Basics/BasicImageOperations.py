#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[3]:


# read image
img = cv2.imread('lolaandstitch.jpeg')


# In[10]:


# show image
cv2.imshow("Image", img)
cv2.waitKey(0) # to wait until a click occurs
cv2.destroyAllWindows() # to prevent any kind of problem we should take care of this line all the time


# In[11]:


# show image in grayscale
img1 = cv2.imread('lolaandstitch.jpeg', cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE = 0
cv2.imshow("Image", img1)
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[9]:


print(img)


# In[12]:


# to change the size of the window
cv2.namedWindow("LolaAndStitch", cv2.WINDOW_NORMAL) # the name of the window, window normal ensure us to size of the window
cv2.imshow("LolaAndStitch", img)
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[13]:


# save the image
cv2.namedWindow("LolaAndStitch", cv2.WINDOW_NORMAL) # the name of the window, window normal ensure us to size of the window
cv2.imshow("LolaAndStitch", img)
cv2.imwrite('Test.jpeg', img)
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[19]:


cv2.namedWindow("LolaAndStitch", cv2.WINDOW_NORMAL) # the name of the window, window normal ensure us to size of the window
cv2.imshow("LolaAndStitch", img)
cv2.imwrite("C:\\Users\\S BASA\\Desktop\\ComputerVision(CV2)\\CV2Basics\\Test.jpeg",img) # to save the file to the desktop directory
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[26]:


# resizing the window
cv2.namedWindow('LolaAndStitch')
img = cv2.resize(img, (1256, 560))
cv2.imshow("LolaAndStitch", img)
cv2.imwrite('Test.jpeg', img)
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[27]:


# aspect ratio
def resizewithAspectRatio(img, width, height, inter=cv2.INTER_AREA):
    dimension = None
    (h, w) = img.shape[:2]
    
    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dimension = (int(w*r), height)
    else:
        r = width / float(w)
        dimension (width, int(h*r))
    return cv2.resize(img, dimension, interpolation=inter)


# In[31]:


img_resized = resizewithAspectRatio(img, width=None, height=300, inter=cv2.INTER_AREA)
cv2.imshow("LolaAndStitchResized", img_resized)
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[ ]:




