# Capstone

To run face detect, type:
"python FaceDetect/webcam.py"

To run pattern detect, type:
"python PatternDetect/simpleFaceObscure.py"

"Python" may need to be replaced by "py" to run the programs (Jeff needs to do this, not sure if others do)

Command line arguments:
-r, --record      Records the webcam input and saves to a file in the folder "recordings"
-v, --video       Inputs a video to be run thru the detection. Input as a string (not working currently)
-ui               Removes all UI features (detection boxes, text overlay, etc) for recording

To quit, press q

Also includes Aruco generation now.

May need to install opencv-contrib-python to use aruco:
pip install opencv-contrib-python
