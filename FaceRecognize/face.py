import face_recognition
import os
import cv2

KNOWN_FACES_DIR = '/home/andrew/Work/face_recognition/known_faces'
UNKNOWN_FACES_DIR = '/home/andrew/Work/face_recognition/unknown_faces'
PREDICTION_DIR = '/home/andrew/Work/face_recognition/prediction'
TOLERANCE = 0.6  # 0-1 to tweak sensitivity of labelling images -default is 0.6 lower = more strict
FRAME_THICKNESS = 3  # bbox (bounding box) width
FONT_THICKNESS = 2  #
MODEL = 'cnn'  # convolutional neural networks a CUDA accelerated model!!
# MODEL = 'hog' # histogram of oriented gradients

known_faces = []
known_names = []

# iterate over known faces directory and load
for name in os.listdir(KNOWN_FACES_DIR):
    # load every file of faces from each known person
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        #load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')  # file path
        encoding = face_recognition.face_encodings(image)[0]  # encode the image [0] -> first one it finds
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