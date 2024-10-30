# installed cvzone (Computer Vision Zone), which is a computer vision helping library
# installed mediapipe, which is used for machine learning solutions and applications
# installed tensorflow, which is an open source machine learning framework
#  tensorflow is version 2.9.1

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier  # this will help us to classify signs
import numpy as np  # this is to create a matrix
import math
import time

# build the webcam

cap = cv2.VideoCapture(0)  # this is the capture object
# 0 is the id number for the cam

detector = HandDetector(maxHands=1)  # we only want to detect 1 hand for our data collection
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

offset = 20  # this just makes the box slightly bigger so there's no cutoff
imgSize = 300  # the fixed image size of the hands

folder = "data/C"
counter = 0  # keeps track of how many images have been saved

labels = ["A", "B", "C", "D", "E", "F"]
# using the index we can return the proper label
# for example, if the returned index from classifier.getPrediction() is 0, it will return "A"

# this loop runs the webcam
while True:
    try:
        success, img = cap.read()
        imgOutput = img.copy()
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
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w  # k is our constant
                heightCalculated = math.ceil(k * h)  # rounds up if it's a decimal
                imgResize = cv2.resize(imgCrop, (imgSize, heightCalculated))
                imgResizeShape = imgResize.shape

                # center the cropped image in the white image
                heightGap = math.ceil((imgSize - heightCalculated) / 2)

                imgWhite[heightGap: heightCalculated + heightGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            # to make the letter more visible, add another rectangle
            cv2.rectangle(imgOutput, (x - offset, y - offset - 90), (x - offset + 50, y - offset), (255, 255, 255), cv2.FILLED)

            cv2.putText(imgOutput, labels[index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 1.75, (255,0, 255), 2)
            # output, letter, coordinate, font, size, color, thickness

            cv2.rectangle(imgOutput, (x - offset, y - offset - 20), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            #  we want to send the imgWhite to the model, which will give us a classification
    except:
        cv2.putText(imgOutput, "Move hand", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.75, (0, 0, 255), 2)
        # if a hand is detected but it's cropped

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)  # gives 1 millisecond delay
