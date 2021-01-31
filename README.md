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
-b, --blur        Enables blurring instead of black box obscuring
--hide            Hides the output window (if you just want to process videos)


To finish processing queue, press w
To quit and lose the queue, press q

Also includes Aruco generation now.

Need to install the following:

pip install opencv-contrib-python
pip install ffmpeg
pip install ffmpeg-python


From https://github.com/BtbN/FFmpeg-Builds/releases
Download:
ffmpeg-n4.3.1-29-g89daac5fe2-win64-gpl-4.3.zip
And unzip it
(I unzipped it to c:/ffmpeg/ instead of the default)
and add
c:/ffmpeg/bin/
to the PATH after unzipping
(and restart your code environment after)

Sample command line inputs:
python PatternDetect/simpleFaceObscure.py -r -n "test_recording_1" -ui
^ Records a video without face detecting enabled
python PatternDetect/simpleFaceObscure.py -v "test_recording_1"
^ Watch a video with the pattern detection being applied to it
python PatternDetect/simpleFaceObscure.py -v "input_recording.avi" -r -n "output_recording.avi"
^ Take an input video and apply pattern detection to it, then save the output as a new video


