#import libraries
import cv2
import numpy as np
import os
import mediapipe as mp

#import modules
from proposture.utils import load_video, get_angles, get_landmarks, get_video_dimensions, get_sideview
from proposture.visuals import show_status
from proposture.metrics import get_reps_and_stage, get_neck
from proposture.visuals import show_neck

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#load video
video_file_path = "../media/full_pushup.mp4"
cap = load_video(video_file_path)
height, width = get_video_dimensions(cap)
advice_list = []

#set up mediapipe instance
# TODO: set view variable (can be passed as *args)
def main(cap, height, width, view = 'front', rep_counter = 0, stage = 'START'):
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

            #if sideview, get sideview angle
            if view == 'side':
                sideview_angle = get_sideview(*landmarks)

            # Calculate angles of joints
            angles = get_angles(*landmarks)
            # Name joint angle variables
            elbow_angles = angles[:2]
            neck_angles = angles[2:4]
            wrist_angles = angles[4:6]
            shoulder_angles = angles[6:8]
            hip_angles = angles[8:10]
            knee_angles = angles[10:12]

            #get status box
            reps_stage = get_reps_and_stage(elbow_angles, rep_counter, stage)
            show_status(image, *reps_stage)
            #update stage and rep_counter for next loop
            stage = reps_stage[0]
            rep_counter = reps_stage[1]

            #get status and advice
            # TODO: based on view, implement different get_metrics function
            if view == 'side':
                neck = get_neck(neck_angles, sideview_angle)

            # TODO: based on view, implement different get_metrics function
            # if view == 'front':

            # Visualize status on joints
            neck_status = show_neck(image, neck, sideview_angle, height, width, *landmarks)
            neck_status
            if not (neck[1] in advice_list):
                    advice_list.append(neck[1])

            # Display advice text
            if len(advice_list) >0:
                cv2.rectangle(image, (0, 80), (400,150), (36, 237, 227), -1)
                # advice text
                cv2.putText(image, "ADVICE:", (15,100),
                        cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, advice_list[0], (15,130),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                if len(advice_list)>1:
                    cv2.rectangle(image, (0, 150), (400,180), (36, 237, 227), -1)
                    cv2.putText(image, advice_list[1], (15,170),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                    if len(advice_list)>2:
                        cv2.putText(image, advice_list[2], (15,190),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

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
    # video_file_path = "../media/full_pushup.mp4"
    # cap = load_video(video_file_path)
    # height, width = get_video_dimensions(cap)
    main(cap, height, width, view= 'side')
