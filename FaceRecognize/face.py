import face_recognition
import os
import cv2
import numpy as np
from os.path import isfile, join

#KNOWN_FACES_DIR = '/home/andrew/Work/face_recognition/known_faces'
#UNKNOWN_FACES_DIR = '/home/andrew/Work/face_recognition/unknown_faces'
#UNKNOWN_FACES_VID_DIR = '/home/andrew/Work/face_recognition/unknown_faces/videos'
#PREDICTION_DIR = '/home/andrew/Work/face_recognition/prediction'

KNOWN_FACES_DIR = '/home/andrew/Work/capstone/FaceRecognize/known_faces'
UNKNOWN_FACES_DIR = '/home/andrew/Work/capstone/FaceRecognize/unknown_faces'
UNKNOWN_FACES_VID_DIR = '/home/andrew/Work/capstone/FaceRecognize/unknown_faces/videos'
PREDICTION_DIR = '/home/andrew/Work/capstone/FaceRecognize/prediction'

TOLERANCE = 0.6  # 0-1 to tweak sensitivity of labelling images -default is 0.6 lower = more strict
FRAME_THICKNESS = 3  # bbox (bounding box) width
FONT_THICKNESS = 2  #
MODEL = 'cnn'  # convolutional neural networks a CUDA accelerated model!!
# MODEL = 'hog' # histogram of oriented gradients

known_faces = []
known_names = []


def get_frame(sec, name, vidcap, count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    
    hasFrames, image = vidcap.read()
    if hasFrames:
        print("writing image")
        save_name = str(count) + name + ".jpg"
        cv2.imwrite(os.path.join(UNKNOWN_FACES_DIR, save_name), image)  # save each frame as a jpg file
    return hasFrames

# Take input videos and generate images to be tested on
videos = []
for filename in os.listdir(UNKNOWN_FACES_VID_DIR):
    if filename.endswith(".mp4") or filename.endswith(".mov") or filename.endswith(".wmv"):  # if the file is a video
        videos.append(os.path.join(UNKNOWN_FACES_VID_DIR, filename))
        print("converting video to image: "+ os.path.join(UNKNOWN_FACES_VID_DIR, filename))
        vidcap = cv2.VideoCapture(os.path.join(UNKNOWN_FACES_VID_DIR, filename))

        success = True
        sec = 0
        frameRate = 5  # captures each 5 second of the video
        count = 1
        while success:  # now we have video, capture images from it
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = get_frame(sec, filename, vidcap, count)
    else:
        continue


# iterate over known faces directory and load
for name in os.listdir(KNOWN_FACES_DIR):
    # load every file of faces from each known person
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        #load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')  # file path
        encoding = face_recognition.face_encodings(image)[0] # encode the image [0] -> first face it finds
        known_faces.append(encoding)
        known_names.append(name)

print("processing unknown faces")
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model=MODEL)  # face detection
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)  # known faces to the encoding we are doing
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"match found: {match}")  # we found a match, now draw a bbox

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)  # for text label
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            # draw rectangle and display match label
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)
    cv2.imshow(filename, image)
    cv2.imwrite((f"{PREDICTION_DIR}/{filename}.jpg"),image)
    cv2.waitKey(5000) # 5 seconds
    #cv2.destroyWindow(filename)  # opencv on ubuntu doesn't like this line
    