import cv2
import imutils
import argparse
import numpy as np
from skimage.filters import threshold_local

from utils import four_point_transform

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-i", "--image", required = True,
        help = "Path to the image for scanning")
args = vars(argument_parser.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

cv2.imshow("Edged", edged)
cv2.waitKey(0)

for contour in cnts:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    print(approx)

    if len(approx) >= 4:
        screenCnt = approx
        break
    else:
        print("cannot scan image reliably")
        exit(-1)

cv2.drawContours(image, [screenCnt], -1, (255, 0, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))


cv2.waitKey(0)
cv2.destroyAllWindows()
