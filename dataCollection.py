import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np  # this is to create a matrix
import math
import time


# build the webcam

cap = cv2.VideoCapture(0)  # this is the capture object
# 0 is the id number for the cam

detector = HandDetector(maxHands=1)  # we only want to detect 1 hand for our data collection

offset = 20  # this just makes the box slightly bigger so there's no cutoff
imgSize = 300  # the fixed image size of the hands

folder = "data/I"
counter = 0  # keeps track of how many images have been saved

# this loop runs the webcam
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    # we have to crop the image
    if hands:
        hand = hands[0]  # we can index at 0 because there's only 1 hand
        x, y, w, h = hand["bbox"]  # index the dictionary using key "bounding box"

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        # this is a square matrix of ones, 3 shows that its colored
        #  np.uint8 is because since it's colored, values are [0,255]

        #  create a try-except to account for image size, it can't be too big or it crashes

        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]
        # this gives us the exact bounding box required
        # y is the starting height, y + h is ending height
        # x is the starting width, x + h is the ending width

        #  we want to overlay the cropped image and fixed size white image
        imgCropShape = imgCrop.shape


        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize / h  # k is our constant
            widthCalculated = math.ceil(k * w)  # rounds up if it's a decimal
            imgResize = cv2.resize(imgCrop, (widthCalculated, imgSize))
            imgResizeShape = imgResize.shape

            # center the cropped image in the white image
            widthGap = math.ceil((imgSize - widthCalculated) / 2)

            imgWhite[:, widthGap: widthCalculated + widthGap] = imgResize
        else:
            k = imgSize / w  # k is our constant
            heightCalculated = math.ceil(k * h)  # rounds up if it's a decimal
            imgResize = cv2.resize(imgCrop, (imgSize, heightCalculated))
            imgResizeShape = imgResize.shape

            # center the cropped image in the white image
            heightGap = math.ceil((imgSize - heightCalculated) / 2)

            imgWhite[heightGap: heightCalculated + heightGap, :] = imgResize



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        #  if height > width, set height to 300, calculate width, stretch width, vice versa
        #  after calculations, put new stretched image in the center of white image

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)  # gives 1 millisecond delay
    if key == ord("s"):  # if the user clicks "s" on the keyboard, we want to save the image
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(counter)
    #  we want to save about 300 images for each letter of the ASL alphabet

    #  after collecting the images, we need to go to Google teachable machine
    #  this allows for the machine to be trained on the images
    #  select image project
    #  select standard image model
    #  add your classes, each class is a letter
    #  after you have trained the model, export the model
    #  choose tensorflow and keras, then download the model
    #  create a new folder called "model" and copy the downloaded model there


