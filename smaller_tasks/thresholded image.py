# Script for thresholding.
import cv2
import numpy as n
import time 

# Load the two images to grayscale, use ie cooler bottle for the task.
scene1 = cv2.imread('img1.jpeg')
scene1_gray = cv2.cvtColor(scene1, cv2.COLOR_BGR2GRAY)

scene2 = cv2.imread('img2.jpeg')
scene2_gray = cv2.cvtColor(scene2, cv2.COLOR_BGR2GRAY)

# Take the absolute value difference between the images
diff = cv2.absdiff(scene1_gray, scene2_gray)

# Remove the noise with binary thresholding.
_, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

# Press 0 twice to see both of the images
cv2.imshow('Difference', diff)
cv2.waitKey(0)
cv2.imshow('Thresholded', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()






