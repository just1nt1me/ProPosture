'''This file outputs advice based on metrics.py'''
import cv2
from metrics import get_reps_and_stage

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
