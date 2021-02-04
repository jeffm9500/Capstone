import sys
import cv2
import numpy as np
import os
import time
import argparse
import ffmpeg


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
    return (x_diff**2.5+y_diff)

def getFaceCentres(faces):
    faceCentres=[]
    for (x, y, w, h) in faces:
            faceCentres.append(np.array([x+w/2,y+h/2]))   
    return faceCentres 


def getBlockedFaces(corners, faces):

    """ TODO: 
    - have the geometric box scale according to the size of the corners
    - figure out the optimal value for minValidScore (min distance score before geometric approach enabled)
    - generalize pixel values to ratios instead of absolute values in geometric approach
    """

    minValidScore = 6500
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

        #print(f'Faces: {faces[np.argmin(faceScore)]}')


        # Append the face with the nearest centre to the new array
        if min(faceScore) < minValidScore:
            blockedFaces.append(faces[np.argmin(faceScore)])
        else:
            blockedFaces.append([int(markerCentre[0]-70), int(markerCentre[1]-300),150,250])
            #print(blockedFaces)
    return blockedFaces

def checkLostFaces(prevFaces, faces, w_dim, h_dim):
    #total number is different AND the inside number is different
    if len(prevFaces) != len(faces):
        pass
        return False
    else:
        return False

# Code from https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting
def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    try:
        meta_dict = ffmpeg.probe(path_video_file)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    

    rotateCode = None
    # Check extension name:
    
    if path_video_file.lower().endswith('.mov'):
        rotate = meta_dict.get('streams', [dict(tags=dict())])[1].get('tags', dict()).get('rotate', 0)
        rotateCode = round(int(rotate) / 90.0) * 90
    else:
        rotate = meta_dict.get('streams', [dict(tags=dict())])[0].get('tags', dict()).get('rotate', 0)
        rotateCode = round(int(rotate) / 90.0) * 90

    print(f'Rotate Code:{rotateCode}')
    return rotateCode

def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 


