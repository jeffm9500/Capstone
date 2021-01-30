import face_recognition
import os
import cv2

KNOWN_FACES_DIR = '/home/andrew/Work/capstone/FaceRecognize/known_faces'
#UNKNOWN_FACES_DIR = '/home/andrew/Work/capstone/FaceRecognize/unknown_faces'
UNKNOWN_FACES_VID_DIR = '/home/andrew/Work/capstone/FaceRecognize/unknown_faces/videos'
PREDICTION_DIR = '/home/andrew/Work/capstone/FaceRecognize/prediction'

TOLERANCE = 0.6  # 0-1 to tweak sensitivity of labelling images -default is 0.6 lower = more strict
FRAME_THICKNESS = 3  # bbox (bounding box) width
FONT_THICKNESS = 2  #
#MODEL = 'cnn'  # convolutional neural networks a CUDA accelerated model!!
MODEL = 'hog' # histogram of oriented gradients -- using this because its faster than CNN

#video = cv2.VideoCapture(0)  # this is the webcam, can also be a filename
video = cv2.VideoCapture("video.mp4")

frame_num = 0
total_frames_num = 0  # how many frames we looked at
hits = 0  # a measure to track how many times we are able to see faces
img_array = []

# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


print('Loading known faces...')
known_faces = []
known_names = []

# Take input videos and generate images to be tested on
for name in os.listdir(KNOWN_FACES_DIR):
    # Next we load every file of faces of known person
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]
        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)


print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
while True:

    # Load image
    #print(f'Filename {filename}', end='')
    #image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')

    ret, image = video.read()
    total_frames_num = total_frames_num + 1
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f', found {len(encodings)} face(s)')
    hits = hits + len(encodings)  # increment for amount of faces found (can also be zero)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = name_to_color(match)
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    cv2.imshow(filename, image)
    frame_name = str(frame_num) + ".jpg"
    cv2.imwrite(f"{PREDICTION_DIR}/{frame_name}", image)  # write all the frames to prediction directories
    frame_num = frame_num + 1  # increment for the next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # if user presses the q key
        break
    #cv2.waitKey(5000)
    #cv2.destroyAllWindows()
    height, width, layers = image.shape
    size = (width, height)
    img_array.append(image)

hits_ratio = hits / total_frames_num
print("Ratio of hits to frames is: " + str(hits_ratio))

video_out_path = PREDICTION_DIR + "/output.avi"
# videowriter(path, type, fps, size of frame)
out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'DIVX'), 5, size) 

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()