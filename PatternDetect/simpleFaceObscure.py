import cv2
import sys
import cv2.aruco as aruco
import numpy as np
import argparse
from time import strftime
from datetime import datetime



def thresholdScore(x_diff, y_diff):
    return (x_diff**2+y_diff)

def getBlockedFaces(corners, faces):
    blockedFaces = []
    faceCentres = []

    for (x, y, w, h) in faces:
        faceCentres.append(np.array([x+w/2,y+h/2]))

    # Loop through all the markers
    for marker in corners:

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

extension = ".avi"
now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H-%M-%S")
out_video_name = current_time + extension
out_folder = "recordings/"

# Parsing command-line arguments:
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", action='store', help="Video File Path", required=False)
parser.add_argument("-r", "--record", action='store_true', help="Enable recording", required=False)
parser.add_argument("-ui", action='store_true', help="Disable all features, leaving just the raw webcam", required=False)

args = parser.parse_args()

if args.video:
    # Set input to be the video defined in the command line argument
    print("Video feed input from: ", args.video)
    video_capture = cv2.VideoCapture(args.video)

else:
    # Default to webcam as the input
    print("No video feed inputted, using webcam as input")
    video_capture = cv2.VideoCapture(0)
    
if args.record:
    # Set the webcam to record, and output the recording 
    print("Recording webcam and saving to file: ", out_video_name)
    #video_capture = cv2.VideoCapture(0)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

profileCascPath = "haarcascade_profileface.xml"
profileFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + profileCascPath)



# Gather width and height of input webcam
w_dim = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h_dim = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
if args.record:
    video_output = cv2.VideoWriter(out_folder + out_video_name, fourcc, 30.0, (w_dim,h_dim))


#video_output = cv2.VideoWriter('test.avi', fourcc, 20.0, (w_dim,h_dim))

# ArUco parameter preparation
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

# have a delay if temporary glitch in ArUco detection
detectCount = 60

# How many frames to wait without marker to continue obscuring face
detectDelay = 17

print("debug:")
print(args.video)
print(video_capture.isOpened())

#while True:
while not args.video or video_capture.isOpened():

    """
    # Check if the input video has ended
    if (args.video and not video_capture.isOpened()):
        print("end of video")
        break
"""
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # If -ui is enabled, then only the raw webcam should be recorded
    if args.ui:
        # Don't add any elements to the frame, just the raw footage
        if args.record:
            video_output.write(frame)

    else:
        # Continue adding UI elements as normal

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = []
        # Detect front faces
        faces_straight = faceCascade.detectMultiScale(
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

        for f in faces_straight:
            faces.append(f)
        for f in faces_profile_right:
            faces.append(f)
        for f in faces_profile_left:
            faces.append(f)



        # Display the resulting frame
        
            # Lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if detectCount<detectDelay or ids is not None:
            
            
            #blockedFaces = getBlockedFaces(corners[0], faces)
            # getBlockedFaces returns array of coordinates for faces to block

            #print(f'Detecting ArUco with ID:{ids}')
            if ids is None:
                detectCount +=1
                cv2.putText(frame, 'No marker - maintaining block', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                # In here must make a function that chooses the face with the centre nearest to the previously blocked face
            elif faces:
                detectCount = 0
                blockedFaces = getBlockedFaces(corners[0], faces)
                cv2.putText(frame, 'Marker detected - blocking', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            for (x, y, w, h) in blockedFaces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)
            """
                # Draw a rectangle around right profile faces (light blue)
            for (x, y, w, h) in faces_profile_right:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)

                # Draw a rectangle around left profile faces (dark blue)
            for (x, y, w, h) in faces_profile_left:
                cv2.rectangle(frame, (w_dim-x, y), (w_dim-x-w, y+h), (0, 0, 0), -1)
            """
        else:
            cv2.putText(frame, 'No marker - showing faces', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        
        if args.record:
            video_output.write(frame)

    # Show frame

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
if args.record:
    video_output.release()
cv2.destroyAllWindows()