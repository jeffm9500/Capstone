import numpy as np
import cv2
import cv2.aruco as aruco

"""
Script uses cv2 built-in ArUco detection to detect an ArUco marker, draw a box around it, and print the ID of the marker in the console.

"""

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()

    # Lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None:
        print(f'Detecting ArUco with ID:{ids}')

    # Draw region around the ArUco box
    frame = aruco.drawDetectedMarkers(frame, corners)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()