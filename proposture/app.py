#import libraries
import cv2
import numpy as np
import os
import mediapipe as mp
from data import load_video, get_dimensions, get_angles, get_landmarks
from visuals import show_status
from metrics import get_reps_and_stage

#import modules
from data import video_dim

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#load video
cap = load_video()

#set up mediapipe instance
# TODO: set view variable (can be passed as *args)
def main(cap, view = 'front'):
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Break the loop if there are no more frames in the video
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Get coordinates for joints
            landmarks = get_landmarks(results)
            # Calculate angles of joints
            angles = get_angles(*landmarks)


            #get status box
            reps_stage = get_reps_and_stage(*angles)
            show_status(image, *reps_stage)

            #get quality
            # if view == 'side':
            #get advice

            #Render detections
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                  mp.solutions.pose.POSE_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 117, 66), thickness=2, circle_radius=2),
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                                  )


            #5 DISPLAYING WINDOW
            cv2.imshow('Mediapipe Feed', image)

            #6 EXITING SHORTCUT
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        #CLOSING DISPLAY WINDOW
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(cap)
