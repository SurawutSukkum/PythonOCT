import cv2
import pytesseract
from pypylon import pylon
from pypylon_opencv_viewer import BaslerOpenCVViewer
from pytesseract import Output
import re
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# org
org = (10, 10)
# font
font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 0.4

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 1
# Pypylon get camera by serial number
serial_number = '23610391'
info = None
for i in pylon.TlFactory.GetInstance().EnumerateDevices():
    if i.GetSerialNumber() == serial_number:
        info = i
        print('Camera found')
        break
else:
    print('Camera with {} serial number not found'.format(serial_number))

# VERY IMPORTANT STEP! To use Basler PyPylon OpenCV viewer you have to call .Open() method on you camera
if info is not None:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
    camera.Open()


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

img = cv2.imread(r'Picture1.jpg')




y1=0
x1=0
h1=50
w1=400
crop_img1 = img[y1:y1+h1, x1:x1+w1]

y2=90
x2=300
h2=30
w2=170
crop_img2 = img[y2:y2+h2, x2:x2+w2]


y3=160
x3=290
h3=30
w3=150
crop_img3 = img[y3:y3+h3, x3:x3+w3]

y4=130
x4=60
h4=20
w4=120
crop_img4 = img[y4:y4+h4, x4:x4+w4]

y5=150
x5=60
h5=20
w5=120
crop_img5 = img[y5:y5+h5, x5:x5+w5]

y6=210
x6=40
h6=20
w6=120
crop_img6 = img[y6:y6+h6, x6:x6+w6]

y7=225
x7=50
h7=20
w7=160
crop_img7 = img[y7:y7+h7, x7:x7+w7]
cv2.imshow('crop_img7', crop_img7)

y8=240
x8=50
h8=20
w8=170
crop_img8 = img[y8:y8+h8, x8:x8+w8]
cv2.imshow('crop_img8', crop_img8)

gray = get_grayscale(img)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img1, config=custom_config)
print('Read result= ',text)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img2, config=custom_config)
print('Read result= ',text)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img3, config=custom_config)
print('Read result= ',text)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img4, config=custom_config)
print('Read result= ',text)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img5, config=custom_config)
print('Read result= ',text)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img6, config=custom_config)
print('Read result= ',text)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img7, config=custom_config)
print('Read result= ',text)

# Adding custom options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(crop_img8, config=custom_config)
print('Read result= ',text)

cv2.imshow('img', img)
cv2.waitKey(0)

'''

# demonstrate some feature access
new_width = camera.Width.GetValue() - camera.Width.GetInc()
if new_width >= camera.Width.GetMin():
    camera.Width.SetValue(new_width)

numberOfImagesToGrab = 100
camera.StartGrabbingMax(numberOfImagesToGrab)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data.
        print("SizeX: ", grabResult.Width)
        print("SizeY: ", grabResult.Height)
        img = grabResult.Array
        print("Gray value of first pixel: ", img[0, 0])

    grabResult.Release()
camera.Close()
'''
