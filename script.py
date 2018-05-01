import numpy
import cv2
import sys

# load the image and convert it to grayscale
image = cv2.imread('images/' + sys.argv[1] + '.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# sobel filter
gradientX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradientY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

# substract the y-gradient from the x-gradient
gradient = cv2.subtract(gradientX, gradientY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)


(_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = numpy.int0(cv2.boxPoints(rect))

# draw a border box arounded the detected barcode
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

# show the image with the signed area in which the barcode is detected
cv2.imshow("Image", image)

cv2.waitKey(0)