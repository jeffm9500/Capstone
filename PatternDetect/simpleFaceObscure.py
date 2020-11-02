import cv2
import sys
import cv2.aruco as aruco
import numpy as np


# Credit: alpython.com/blog/python/face-detection-in-python-using-a-webcam/

# Slightly modified by Jeff
# Press q to quit the window

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

profileCascPath = "haarcascade_profileface.xml"
profileFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + profileCascPath)

video_capture = cv2.VideoCapture(0)

# Gather width and height of input webcam
w_dim = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h_dim = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ArUco parameter preparation
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

# have a delay if temporary glitch in ArUco detection
detectCount = 60

# How many frames to wait without marker to continue obscuring face
detectDelay = 17

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect front faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    
    # Detect right-turned profile faces
    faces_profile_right = profileFaceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )



    # Flip image for left-profile detection
    frame_swap = cv2.flip(frame, 1)
    gray_swap = cv2.cvtColor(frame_swap, cv2.COLOR_BGR2GRAY)
    
    # Detect left-turned profile faces
    faces_profile_left = profileFaceCascade.detectMultiScale(
        gray_swap,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    # Display the resulting frame
    
        # Lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if detectCount<detectDelay or ids is not None:
        print(f'Detecting ArUco with ID:{ids}')
        if ids is None:
            detectCount +=1
            cv2.putText(frame, 'No marker - maintaining block', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            detectCount = 0
            cv2.putText(frame, 'Marker detected - blocking', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)

            # Draw a rectangle around right profile faces (light blue)
        for (x, y, w, h) in faces_profile_right:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)

            # Draw a rectangle around left profile faces (dark blue)
        for (x, y, w, h) in faces_profile_left:
            cv2.rectangle(frame, (w_dim-x, y), (w_dim-x-w, y+h), (0, 0, 0), -1)

    else:
        cv2.putText(frame, 'No marker - showing faces', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    
    cv2.imshow('Video', frame)

    # "q" key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    # "esc" key pressed
    if cv2.waitKey(1) & 0xFF == ord('\x1b'):
        break
    # ^ this doesnt work yet btw


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
