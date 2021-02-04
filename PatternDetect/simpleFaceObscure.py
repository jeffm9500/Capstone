import cv2
import sys
import cv2.aruco as aruco
import numpy as np
import argparse
import imutils
import time
from time import strftime
from datetime import datetime
from helperFunctions import *
from FileVideoStream import FileVideoStream
from imutils.video import FPS
#from imutils.video import FileVideoStream
import ffmpeg





t = time.time()


# Parsing command-line arguments:
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", action='store', help="Video File Path (include extension)", required=False)
parser.add_argument("-r", "--record", action='store_true', help="Enable recording", required=False)
parser.add_argument("-ui", "--userint", action='store_true', help="Disable all features, leaving just the raw webcam", required=False)
parser.add_argument("-n", "--name", action='store', help="Provide a file name to store recording as (include extension)", required=False)
parser.add_argument("-b", "--blur", action='store_true', help="Enable blurring instead of black box obscuring", required=False)
parser.add_argument("--hide", action='store_true', help="Hide the output window", required=False)

args = parser.parse_args()

# Setting the recording's file name
extension = ".avi"
if args.name:
    # Set the recording's name instead of the default date/time
    out_video_name = args.name #+ extension
else:
    # Set the recording's name to be the current date/time
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H-%M-%S")
    out_video_name = current_time + extension
out_folder = "recordings/"
in_folder = "recordings/"

if args.video:
    # Set input to be the video defined in the command line argument
    in_video_name = args.video #+ extension
    print(f"Video feed input from: {in_folder}{in_video_name}")

    # check if video requires rotation
    rotateCode = check_rotation("recordings\\" + in_video_name)

    print(f'Starting video file thread...')
    # check if the frame needs to be rotated
    if rotateCode is not None and rotateCode != 0:
        #frame = correct_rotation(frame, rotateCode)
        fvs = FileVideoStream("recordings\\" + in_video_name, rotate=True, rotateCode=rotateCode).start()
    else:
        fvs = FileVideoStream("recordings\\" + in_video_name).start()
    time.sleep(1.0)
    """
    video_capture = cv2.VideoCapture(in_folder + in_video_name)
    if(video_capture.isOpened() == False):
        print(f'Error opening video from: {in_folder}{in_video_name}')
    """
    

else:
    # Default to webcam as the input
    print("No video feed inputted, using webcam as input")
    # potentially can call fvs = FileVideoStream(0).start()
    
    fvs = FileVideoStream(path=0, backend=cv2.CAP_DSHOW).start()
    #
    in_video_name = "No input"  
    # Assuming webcam does not rotate
    #rotateCode = None
    
if args.record:
        
    # Set the webcam to record, and output the recording 
    print(f"Recording and saving to file at: {out_folder}{out_video_name}")

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

profileCascPath = "haarcascade_profileface.xml"
profileFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + profileCascPath)


# Gather width and height of input webcam
w_dim, h_dim = fvs.getDim()


fourcc = cv2.VideoWriter_fourcc(*'XVID')
if args.record:
    video_output = cv2.VideoWriter(filename=out_folder + out_video_name, fourcc=fourcc, fps=30.0, frameSize=(w_dim,h_dim))


print("debug:")
print("in_video_name: " +str(in_video_name))
print("out_video_name: " +str(out_video_name))
print("input open: ", fvs.running())
if args.userint:
    print("ui disabled")
if args.hide:
    print("window hidden")
if args.blur:
    print("blurring enabled")


"""
SCRIPT PARAMETERS
"""
# ArUco parameter preparation
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
parameters =  aruco.DetectorParameters_create()


parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
#parameters.cornerRefinementWinSize = 30
#parameters.cornerRefinementMinAccuracy = 0.15
#parameters.cornerRefinementMaxIterations = 60
#parameters.markerBorderBits = 5

#parameters.polygonalApproxAccuracyRate = 0.03
#parameters.adaptiveThreshWinSizeMax = 3
#parameters.minCornerDistanceRate = 0.02
#parameters.minDistanceToBorder = 2
#parameters.errorCorrectionRate = 1
#parameters.maxErroneousBitsInBorderRate = .9

#parameters.minOtsuStdDev = 0.1

#parameters.cornerRefinementWinSize = 20
#parameters.minDistanceToBorder = 5
#parameters.adaptiveThreshWinSizeMax = 50

# Face detect/obscuring parameters
blockedFaces = []
ids = []
prevNumBlocked = 0
prevFaces = []

# Change this to 1 to switch to blur instead of obstruct
if args.blur:
    blurMode = 1
else:
    blurMode = 0

# How many frames to wait without marker to continue obscuring face
detectDelay = 17

# Initial value to compare to the maximum delay of frames to wait without marker (detectDelay)
detectCount = 18


# Bounds for "in camera view" field
xMin = w_dim*.1
xMax = w_dim*.9
yMin = h_dim*.1
yMax = h_dim*.9

print(f'xMin: {xMin}\nxMax: {xMax}\nyMin: {yMin}\nyMax: {yMax}')
fps = FPS().start()
# Initialize webcam video stream:
#while video_capture.isOpened():
while fvs.more():
    # While loop will quit when queue is empty and flagToStop is True (set with the 's' key)

    # Capture frame-by-frame

    success, frame = fvs.read()
    # Check if the input video has ended
    if not success:
        print("End of video")
        break


    # If -ui is enabled, then only the raw webcam should be recorded
    if args.userint:
        
        # Don't add any elements to the frame, just the raw footage
        if args.record:
            video_output.write(frame)

    else:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        """
        # Continue adding UI elements as normal     
        """
        faces = []
        # Detect front faces
        sf = 1.4
        mn = 4
        ms = (30, 30)
        faces_straight = faceCascade.detectMultiScale(
            gray,
            scaleFactor=sf,
            minNeighbors=mn,
            minSize=ms,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
        # Detect right-turned profile faces
        faces_profile_right = profileFaceCascade.detectMultiScale(
            gray,
            scaleFactor=sf,
            minNeighbors=mn,
            minSize=ms,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Flip image for left-profile detection
        frame_swap = cv2.flip(frame, 1)
        gray_swap = cv2.cvtColor(frame_swap, cv2.COLOR_BGR2GRAY)
        
        # Detect left-turned profile faces
        faces_profile_left = profileFaceCascade.detectMultiScale(
            gray_swap,
            scaleFactor=sf,
            minNeighbors=mn,
            minSize=ms,
            flags=cv2.CASCADE_SCALE_IMAGE
        )


        for f in faces_straight:
            faces.append(f)
        for f in faces_profile_right:
            faces.append(f)
        for f in faces_profile_left:
            # to account for the previous frame flip
            f[0] = w_dim-f[0]-f[2]
            faces.append(f)

        

        # Display the resulting frame
        
        # Lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(0,0,255))
        aruco.drawDetectedMarkers(frame, corners)

        #for i in rejectedImgPoints:
        # Used for displaying rejected contours
        #   corners.append(i)
            #print(f'rejected: {i}')
            #ids.append([1])

        if detectCount<detectDelay or ids is not None:
            if ids is None:
                ids = []
            if len(ids) < prevNumBlocked and detectCount<detectDelay and faces:
                # Continue blocking the same faces from before (case: lost sight of an ID, still within the delay)

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
            

            elif False:
                pass
                # want to make the condition that if a face disappears within the centre region of the frame, continue blocking the area (geometric approach, improves the blocking capacity)
                # define centre region
                    # If the value of the centre of the face that disappears is within the border, keep blocking?

                # OKAY NEW PLAN - if there is not one within the x bounds, then approximate
                
                len(ids) >= prevNumBlocked and checkLostFaces(prevFaces, faces, w_dim, h_dim)


            elif faces:
                # else, the number of faces increases or stays the same
                detectCount = 0
                blockedFaces = getBlockedFaces(corners, faces)
                cv2.putText(frame, 'Marker detected - blocking', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)





            else:
                cv2.putText(frame, 'Markers, but no faces. Geometric approach.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                blockedFaces = getBlockedFaces(corners,[[2000,2000,1000,1000]])



            # Decides if use Gaussian blur or the full obstruct
            if blurMode == 0:
                frame = block(blockedFaces, frame)
            elif blurMode == 1:
                frame = blur(blockedFaces, frame)
            else:
                # Do not block/blur anything
                pass
            

        else:
            cv2.putText(frame, 'No markers - showing faces', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        

        cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if args.record:
            video_output.write(frame)

        prevNumBlocked = len(blockedFaces)
        prevFaces = faces

    # Check if frame should be hidden
    if not args.hide:
        # Show frame
        cv2.imshow('PatternDetect', frame)
        fps.update()

    
    pressedKey = cv2.waitKey(1) & 0xFF
    # "q" key pressed
    if pressedKey == ord('q'):
        print("Stopping")
        break
    # "w" key is pressed
    elif pressedKey == ord('w'):
        # instead of breaking instantly, wait until the queue is processed
        print("Stopping once queue is empty")
        fvs.stop()
        
        

# When everything is done, release the capture
"""
if args.video:
    fvs.release()
else:
    video_capture.release()
"""
fvs.release()
if args.record:
    video_output.release()
fps.stop()
print(f"Elapsed time: {time.time()-t:.2f}s")
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
