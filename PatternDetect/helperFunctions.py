import sys
import cv2
import numpy as np
import os
import time
import argparse

# SOURCE: https://medium.com/@bipinadvani/face-recognition-and-blurring-in-webcam-using-cv2-python-5c4c589e6e59

def blur():
    face = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

    capture = cv2.VideoCapture(0)

    blurred = False
    framed = False

    while True:

        img, frame = capture.read()

        if(img):    # if face is captured
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            blockedFaces = face.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 7)

            for x,y,w,h in blockedFaces:
                # if no blur initialized
                if blurred:
                    # create a sub-face from detected faces
                    sub_face = frame[y:y+h, x:x+w]

                    # apply Gaussian Blur on sub-face
                    sub_face = cv2.GaussianBlur(sub_face,(35,35), 50)   # values for blurring obtained through trial and error

                    # merge rectangle with Gaussian Blur and final image
                    frame[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face

                # initializes a frame around the recognized face
                if framed:
                    cv2.rectangle(frame,(x,y),(x+h,y+w),(255,255,0),2)

            cv2.imshow('Face Recognized', frame)   

        ch = 0xFF &cv2.waitKey(1)

        # initialize blur by pressing "b"
        if ch == ord("b"):
            blurred = not blurred

        # initialize face framing by pressing "f"
        if ch == ord("f"):
            framed = not framed

        # quit frame by pressing "q"
        if ch ==ord("q"):
            break
blur()
