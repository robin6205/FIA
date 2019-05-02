import cv2
import numpy as np

cap = cv2.VideoCapture('C:\\Users\\robin\\PycharmProjects\\test\\Import image\\Intel RealSense Viewer v2.18.1 2019-02-20 01-21-43.mp4')
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('CAnny_Edge_out.avi',fourcc, 20.0, (640,480))

while (1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#Blue filter
    #lower_color = np.array([110, 50, 50])
    #upper_color = np.array([130, 255, 255])

#Yellow Filter
    #lower_color = np.array([25, 50, 50])
    #upper_color = np.array([32, 255, 255])

#color modified Filter
    lower_color = np.array([30,150,50])
    upper_color = np.array([255,255,180])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    res = cv2.bitwise_and(frame, frame, mask=mask)

#Morphological Operations
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    #opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Canny Edge Detection
    mask = cv2.Canny(frame, 100, 200)
#Gaussian Blur
    blur = cv2.GaussianBlur(mask, (15,15), 0)
    cv2.imshow('blur',blur)

#Median Blur
    #median = cv2.medianBlur(mask,15)
    #cv2.imshow('Median Blur',median)

#smoothing
    #kernel = np.ones((15,15),np.float32)/225
    #smoothed = cv2.filter2D(res,-1,kernel)
    #cv2.imshow('Averaging',smoothed)

    # Canny Edge Detection
    mask = cv2.Canny(frame, 100, 200)
    # Gaussian Blur
    blur = cv2.GaussianBlur(mask, (15, 15), 0)
    cv2.imshow('blur', blur)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)


    #cv2.imshow('Erosion',erosion)
    #cv2.imshow('Dilation',dilation)


    #cv2.imshow('Opening',opening)
    #cv2.imshow('Closing',closing)


#    out.write(frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()