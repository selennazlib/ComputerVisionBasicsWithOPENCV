# ComputerVisionBasicsWithCV2

> Computer Vision Basics With opencv is a public repository which contains many project related to operations with cv2


Generated 512x512 (3 rgb) canvas by using np.zeros. It generates a black canvas for us.
![canvas](https://snipboard.io/93Y78g.jpg)

To change the canvas color from black to white we can add 255 to  np.zeros . (0, 0, 0) --> (255, 255, 255) 

![white canvas np.zeros](https://snipboard.io/zG3PwB.jpg)

------------



### Real Time Shape Detection
By using my webcam I detected shapes and also added trackbar to control hsv range. Playing around with the lower and upper color ranges we can actually increase the accuracy . There is an example from this project. I tried the project with a paper which has a pentagon on it. (Also paper has circle shaped holes and we could detect that too.) 

![](https://snipboard.io/Lphvfd.jpg)


### Smile Detection
We know that cascade files are not effective as we desire . However in many situation we can use them to detect faces, eyes, smiles etc.  but we have to be aware of it does not  get enough sufficient to detect an object. Here some examples of smile cascade; 
*Note:* To detect specific parts of a frame(or an image) we can use multiple cascades(xml files). E.g., to detect a smile I both use frontalface.xml and smile.xml , which enhanced the results.


![](https://snipboard.io/PTUAiz.jpg)