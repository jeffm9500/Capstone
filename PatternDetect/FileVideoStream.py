import sys
import cv2
import time
import threading as th
from threading import Thread
import multiprocessing
if sys.version_info >= (3, 0):
    from queue import Queue
else:
    from Queue import Queue

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

profileCascPath = "haarcascade_profileface.xml"
profileFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + profileCascPath)


# Code for multi-threading from:
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
# and:
# https://github.com/jrosebr1/imutils/blob/master/imutils/video/filevideostream.py
class FileVideoStream:
    def __init__(self, path, backend=None, transform=None, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path, backend)
        self.stopped = False
        self.stopping = False
        self.transform = transform
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.w_dim = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h_dim = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self
    
    def update(self):
        # keep looping indefinitely
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                break
            
            # otherwise, ensure the queue has room in it
            if not self.Q.full():

                # read the next frame from the file
                grabbed, frame = self.stream.read()

                if not grabbed:
                    print("End of video - from inside thread")
                    self.stopped = True
                else:
                
                    if self.transform:
                        frame = self.transform(frame)
                    self.Q.put(frame)

            else:
                time.sleep(0.1) #rest for 10ms, queue is full
    
    def read(self):
        try:
            frame = self.Q.get(True, 0.1)
        except queue.Empty:
            print("Queue empty")
            return False, None
        else:
            return True, frame

    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.thread.join()

    def getDim(self):
        return self.w_dim, self.h_dim

    def release(self):
        self.stream.release()

    def stopping(self):
        # set queue to stop accepting new frames
        self.stopping = True




         
    
        

        
