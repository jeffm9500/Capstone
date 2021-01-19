import numpy as np
import cv2
import cv2.aruco as aruco

"""
Script generates an ArUco marker and saves it under test_marker.jpg. The marker is defined from the 250 size aruco.DICT.
"""

# Use this to specify which ID of marker to be genreated. Choose between 1-250.
markerID = 3

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
print(aruco_dict)

# Final parameter in drawMarker is the total image size
img = aruco.drawMarker(aruco_dict, markerID, 700)
cv2.imwrite("test_marker.jpg", img)

cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()