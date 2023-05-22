# Load an image containing a side view of a bicycle withclear lines and circles.
# Use the Hough line and Hough circle functions inOpenCV and detect the lines and circles.
import cv2 
import numpy as np

src = cv2.imread('bicycle.jpeg', cv2.IMREAD_COLOR)
#bycicle_img = cv2.medianBlur(src, 5)
bycicle_img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(bycicle_img_gray, 0, 100, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi/180, 160)

# Draw detected lines on the original image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the output image
cv2.imshow("output", src)
cv2.waitKey(0)

# Set the Hough circle detection parameters
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 200, param1=70, param2=60, minRadius=4, maxRadius=200)

if circles is not None:
   circles = np.round(circles[0, :]).astype("int")
   # loop over the (x, y) coordinates and radius of the circles
   for (x, y, r) in circles:
      # draw the circle in the output image, then draw a rectangle
      # corresponding to the center of the circle
      cv2.circle(src, (x, y), r, (0, 255, 0), 2)
      cv2.rectangle(src, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


cv2.imshow("output", src)
cv2.waitKey(0)
cv2.destroyAllWindows()