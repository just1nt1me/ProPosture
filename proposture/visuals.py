'''This file outputs advice based on metrics.py'''
import cv2
import numpy as np
from metrics import get_reps_and_stage
from utils import get_landmarks, get_video_dimensions

# Variables
# width, height = get_video_dimensions
#landmarks
# landmarks = get_landmarks

#advice variables
advice_shoulder = None # if elbows are too far out
advice_butt = None # if butt is too high
advice_knee = None # if knees are sagging
advice_neck = None # if head is not aligned with back
advice_rep = None # if not full rep
advice_hand = None # if hands are not aligned under shoulders
advice_wrist = None
advice_list = [] # store advice text



#STATUS BOX
def show_status(image, stage, rep_counter):
    #4.1 Setup status box
    status_box = cv2.rectangle(image, (0,0), (250,80), (245,117,16), -1)

    # Rep data
    reps_title = cv2.putText(image, 'REPS', (15,12),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    rep_count_text = cv2.putText(image, str(rep_counter),
                (10,70),
                cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    #Stage data
    stage_title = cv2.putText(image, 'STAGE', (75,12),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    stage_text = cv2.putText(image, stage,
                (60,70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv2.LINE_AA)

    return status_box, reps_title, rep_count_text, stage_title, stage_text



# Determine status
def show_neck(image, neck, sideview_angle, height, width, *landmarks):
    if sideview_angle == 'left':
        neck_status = cv2.circle(image, tuple(np.multiply(landmarks[5], [width, height]).astype(int)),
                        20, neck[0], -1)
    if sideview_angle == 'right':
        neck_status = cv2.circle(image, tuple(np.multiply(landmarks[12], [width, height]).astype(int)),
                        20, neck[0], -1)
    return neck_status

def show_hip(image, hip, sideview_angle, height, width, *landmarks):
    if sideview_angle == 'left':
        hip_status = cv2.circle(image, tuple(np.multiply(landmarks[3], [width, height]).astype(int)),
                        20, hip[0], -1)
    if sideview_angle == 'right':
        hip_status = cv2.circle(image, tuple(np.multiply(landmarks[10], [width, height]).astype(int)),
                        20, hip[0], -1)
    return hip_status

def show_knee(image, knee, sideview_angle, height, width, *landmarks):
    if sideview_angle == 'left':
        knee_status = cv2.circle(image, tuple(np.multiply(landmarks[6], [width, height]).astype(int)),
                        20, knee[0], -1)
    if sideview_angle == 'right':
        knee_status = cv2.circle(image, tuple(np.multiply(landmarks[13], [width, height]).astype(int)),
                        20, knee[0], -1)
    return knee_status

def show_hand(image, hand, sideview_angle, height, width, *landmarks):
    if sideview_angle == 'left':
        hand_status = cv2.circle(image, tuple(np.multiply(landmarks[2], [width, height]).astype(int)),
                        20, hand[0], -1)
    if sideview_angle == 'right':
        hand_status = cv2.circle(image, tuple(np.multiply(landmarks[9], [width, height]).astype(int)),
                        20, hand[0], -1)
    return hand_status


# Get advice
# def get_advice(image, advice_list):
#     if len(advice_list) > 0:
#         cv2.rectangle(image, (0, 80), (400,150), (36, 237, 227), -1)
#         # advice text
#         cv2.putText(image, "ADVICE:", (15,100),
#                 cv2.FONT_HERSHEY_DUPLEX, .5, (0,0,0), 1, cv2.LINE_AA)
#         cv2.putText(image, advice_list[0], (15,130),
#                 cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
#         if len(advice_list)>1:
#             cv2.rectangle(image, (0, 150), (400,180), (36, 237, 227), -1)
#             cv2.putText(image, advice_list[1], (15,170),
#                     cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
#             if len(advice_list)>2:
#                 cv2.putText(image, advice_list[2], (15,190),
#                         cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
#                 if len(advice_list)>3:
#                     cv2.putText(image, advice_list[3], (15,190),
#                             cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
#                     if len(advice_list)>4:
#                         cv2.putText(image, advice_list[4], (15,190),
#                                 cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
#                         if len(advice_list)>5:
#                             cv2.putText(image, advice_list[5], (15,190),
#                                     cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
