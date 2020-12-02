import cv2
import sys
import cv2.aruco as aruco
import numpy as np
import argparse
from time import strftime
from datetime import datetime
from helperFunctions import *




# Parsing command-line arguments:
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", action='store', help="Video File Path", required=False)
parser.add_argument("-r", "--record", action='store_true', help="Enable recording", required=False)
parser.add_argument("-ui", action='store_true', help="Disable all features, leaving just the raw webcam", required=False)
parser.add_argument("-n", "--name", action='store', help="Provide a file name to store recording as", required=False)

args = parser.parse_args()

# Setting the recording's file name
extension = ".avi"
if args.name:
    # Set the recording's name instead of the default date/time
    out_video_name = args.name + extension
else:
    # Set the recording's name to be the current date/time
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H-%M-%S")
    out_video_name = current_time + extension
out_folder = "recordings/"
in_folder = "recordings/"

if args.video:
    # Set input to be the video defined in the command line argument
    in_video_name = args.video + extension
    print("Video feed input from: ", in_folder, in_video_name)
    video_capture = cv2.VideoCapture(in_folder + in_video_name)
    if(video_capture.isOpened() == False):
        print("Error opening video from: ", in_video_folder, in_video_name)

else:
    # Default to webcam as the input
    print("No video feed inputted, using webcam as input")
    video_capture = cv2.VideoCapture(0)
    in_video_name = "No input"  
    
if args.record:
        
    # Set the webcam to record, and output the recording 
    print("Recording webcam and saving to file: ", out_folder, out_video_name)
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


print("debug:")
print("out_video_name: " +str(out_video_name))
print("in_video_name: " +str(in_video_name))
print(video_capture.isOpened())


"""
SCRIPT PARAMETERS
"""
# ArUco parameter preparation
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

# Face detect/obscuring parameters
blockedFaces = []
ids = []
blurMode = 0 # Change this to 1 to switch to blur instead of obstruct

# How many frames to wait without marker to continue obscuring face
detectDelay = 17

# Initial value to compare to the maximum delay of frames to wait without marker (detectDelay)
detectCount = 18



# Initialize webcam video stream:
while video_capture.isOpened():

    
    # Check if the input video has ended
    if (args.video and not video_capture.isOpened()):
        print("end of video")
        break

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
            if ids is None:
                ids = []
            if len(ids) < prevNumBlocked and detectCount<detectDelay and faces:
                # Continue blocking the same faces from before (lost sight of an ID, still within the delay)

                detectCount +=1
                cv2.putText(frame, 'Lost marker - maintaining blocks', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                # In here must make a function that chooses the face with the centre nearest to the previously blocked face

                faceCentres = getFaceCentres(faces)
                # Calculate centres of current faces being detected. Block the centres that are nearest to the blockedFaces centres

                prevBlocked = blockedFaces
                blockedFaces = []
                for (x,y,w,h) in prevBlocked:
                    blockedFaces.append(faces[closestFace(np.array([x+w/2,y+h/2]),faceCentres)])
                
                # find nearest faces in faceCentres
            
            elif faces:
                detectCount = 0
                blockedFaces = getBlockedFaces(corners[0], faces)
                cv2.putText(frame, 'Marker detected - blocking', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)



            # Decides if use Gaussian blur or the full obstruct
            if blurMode == 0:
                frame = block(blockedFaces, frame)
            elif blurMode == 1:
                frame = blur(blockedFaces, frame)
            else:
                # Do not block/blur anything
                pass
            

        else:
            cv2.putText(frame, 'No marker - showing faces', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        if args.record:
            video_output.write(frame)

    # Show frame

    cv2.imshow('PatternDetect', frame)
    prevNumBlocked = len(blockedFaces)

    # "q" key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # "esc" key pressed
    if cv2.waitKey(1) & 0xFF == ord("\x1b"):
        break
    # ^ this doesnt work yet


# When everything is done, release the capture
video_capture.release()
if args.record:
    video_output.release()
cv2.destroyAllWindows()
