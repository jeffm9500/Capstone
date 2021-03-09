import cv2
import sys
import os
import time
from time import strftime


def load_images_from_folder(folder):
    images = []
    imageNames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            imageNames.append(filename)
    if not images:
        print("did not load any images from input folder")
    return images, imageNames

# Takes in input folder located at Capstone/faceDetectTesting/input/, reads all images
# Performs face detection on the set, then reports which images did not have a face detected in them
# and outputs a box drawn on each face in the folder Capstone/faceDetectTesting/output/


t = time.time()


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

profileCascPath = "haarcascade_profileface.xml"
profileFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + profileCascPath)


images, imageNames = load_images_from_folder("faceDetectTesting\\input\\")

report = open("faceDetectTesting\\output\\FaceDetectTesting_report.txt", "w")
report.write("Output for running tests on all images in the images folder\n")

assert len(images) == len(imageNames)
iterations = 0
totalDetections = 0
for idx, image in enumerate(images):

    iterations += 1
    w_dim, h_dim, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = []
    # Detect front faces
    sf = 1.2
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
    frame_swap = cv2.flip(image, 1)
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
    if faces:
        totalDetections += 1

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    result = cv2.imwrite(f'faceDetectTesting/output/out_{imageNames[idx]}', image)
    if result == True:
        print("File saved successfully")
        report.write(f'Found {len(faces)} faces in {imageNames[idx]}\n')
    else:
        print("Error saving file")

    # Stop after x images
    if iterations == 10:
        break

print(f"Elapsed time: {time.time()-t:.2f}s")
report.write("End of test set")
report.write(f'Found {totalDetections} faces in {iterations} images')
print(f'Found {totalDetections} faces in {iterations} images')

