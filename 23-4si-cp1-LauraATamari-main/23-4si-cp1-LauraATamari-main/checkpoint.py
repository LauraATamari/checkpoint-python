import cv2
from matplotlib import pyplot as plt
import numpy as np

#getting image
img = cv2.imread('circulo.png')

#image in grey - better to identify the circles
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#filters the border
edges = cv2.Canny(img_gray,50,150)

#HoughCircles method to identify the circles 
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1.1, minDist=300, param1=300, param2=100, minRadius=60, maxRadius=400)

#copying image in RBG
img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img_rbg,(i[0],i[1]),i[2],(0,0,255),2)
        # draw the center of the circle
        cv2.circle(img_rbg,(i[0],i[1]),2,(255,0,0),10)


#end of segmentation


#starting area and mass center display
cnt = circles[0]
M = cv2.moments(cnt)
print( M )

#calculating coordenates for mass center
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print("centro de massa na possição: ",cx, cy)

#defining font 
font = cv2.FONT_HERSHEY_SIMPLEX
text = cy , cx
origem = (0,50)

#adding text to image
cv2.putText(img_rbg, str(text), origem, font,1,(200,50,0),2,cv2.LINE_AA)

plt.figure(figsize = (10,10))
plt.imshow(img_rbg, cmap="Greys_r", vmin=0, vmax=255); 
plt.show()
