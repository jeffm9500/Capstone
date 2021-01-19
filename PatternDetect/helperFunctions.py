import sys
import cv2
import numpy as np
import os
import time
import argparse


def blur(blockedFaces, inFrame):
    frame = inFrame
    for (x,y,w,h) in blockedFaces:
        # create a sub-face from detected faces
        sub_face = frame[y:y+h, x:x+w]

        # apply Gaussian Blur on sub-face
        sub_face = cv2.GaussianBlur(sub_face,(35,35), 50)   # values for blurring obtained through trial and error

        # merge rectangle with Gaussian Blur and final image
        frame[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face

    return frame

def block(blockedFaces, inFrame):
    frame = inFrame
    for (x, y, w, h) in blockedFaces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)
    return frame

def closestFace(prevCentre, faceCentres):
    faceCentres = np.asarray(faceCentres)
    dist_2 = np.sum((faceCentres - prevCentre)**2, axis=1)
    return np.argmin(dist_2)

def thresholdScore(x_diff, y_diff):
    return (x_diff**2+y_diff)

def getFaceCentres(faces):
    faceCentres=[]
    for (x, y, w, h) in faces:
            faceCentres.append(np.array([x+w/2,y+h/2]))   
    return faceCentres 


def getBlockedFaces(corners, faces):
    blockedFaces = []
    faceCentres = []

    for (x, y, w, h) in faces:
        faceCentres.append(np.array([x+w/2,y+h/2]))

    # Loop through all the markers
    for marker in corners:

        marker = marker[0]
        # get the centre of the marker
        markerCentre = marker.mean(axis=0)

        # create empty array to rank faces
        faceScore = []

        # Loop through all face centres and compare to the marker centres
        for faceCentre in faceCentres:
            faceCentre[0] #x
            faceCentre[1] #y
            
            x_diff = abs(faceCentre[0]-markerCentre[0])
            y_diff = abs(faceCentre[1]-markerCentre[1])
            faceScore.append(thresholdScore(x_diff, y_diff))

        # Append the face with the nearest centre to the new array
        blockedFaces.append(faces[np.argmin(faceScore)])
    return blockedFaces

def checkLostFaces(prevFaces, faces, w_dim, h_dim):
    #total number is different AND the inside number is different
    if len(prevFaces) != len(faces):
        pass
        return False
    else:
        return False
