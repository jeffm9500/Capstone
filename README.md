# Capstone

To run face detect, type:
"python FaceDetect/webcam.py"

To run pattern detect, type:
"python PatternDetect/simpleFaceObscure.py"

"Python" may need to be replaced by "py" to run the programs (Jeff needs to do this, not sure if others do)

Command line arguments:
-r, --record      Records the webcam input and saves to a file in the folder "recordings" (default name is the current date/time)
-n, --name        Adds a specific name to the recording (only works when -r is used as well)
-v, --video       Inputs a video to be run through the detection, input as a string
-ui               Removes all UI features (detection boxes, text overlay, etc) for recording raw footage


To quit, press q

Also includes Aruco generation now.

May need to install opencv-contrib-python to use aruco:
pip install opencv-contrib-python

Sample command line inputs:
python PatternDetect/simpleFaceObscure.py -r -n "test_recording_1" -ui
python PatternDetect/simpleFaceObscure.py -v "recordings/test_recording_1"


