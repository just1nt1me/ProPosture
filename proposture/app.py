#import libraries
import cv2
import numpy as np
import os
import mediapipe as mp

#import modules
from data import video_dim

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#check video dimensions
video_file_path = "../media/full_pushup.mp4"
cap = cv2.VideoCapture(video_file_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width, height

#set up mediapipe instance
def main(cap):
    # Setup mediapipe instance
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                # Break the loop if there are no more frames in the video
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
