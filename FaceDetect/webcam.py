import cv2
import sys

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

    # Draw a rectangle around the faces (green)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Detect right-turned profile faces
    faces_profile_right = profileFaceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around right profile faces (light blue)
    for (x, y, w, h) in faces_profile_right:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

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

    
    

    # Draw a rectangle around left profile faces (dark blue)
    for (x, y, w, h) in faces_profile_left:
        cv2.rectangle(frame, (w_dim-x, y), (w_dim-x-w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
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
