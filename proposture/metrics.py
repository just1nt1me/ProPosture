'''This file takes the angles from data.py and outputs conditions for visuals.py'''
import cv2

# variables for get_reps_and_stage
# stage = 'START' #default is start > DOWN, UP for reps
# rep_counter = 0 # counts number of reps

# Rep Counter and Up/Down Stage
def get_reps_and_stage(elbow_angles, rep_counter, stage):
    left_elbow_angle, right_elbow_angle = elbow_angles
    if left_elbow_angle and right_elbow_angle < 90:
        stage = "DOWN"
    if left_elbow_angle and right_elbow_angle > 110 and stage =='DOWN':
        stage = "UP"
        rep_counter +=1
    return stage, rep_counter

# Uses angles to write posture conditional statements
def get_neck(neck_angles, sideview_angle, advice_neck = None):
    left_neck_angle, right_neck_angle = neck_angles
    if sideview_angle == 'left':
        if left_neck_angle >= 165:
            neck_status_color= (31, 194, 53)
        if left_neck_angle in range(150,165):
            neck_status_color= (0, 255, 255)
        if left_neck_angle < 150:
            neck_status_color= (14, 14, 232)
            advice_neck = 'Tuck chin in'
    if sideview_angle == 'right':
        if right_neck_angle >= 165:
            neck_status_color= (31, 194, 53)
        if right_neck_angle in range(150,165):
            neck_status_color= (0, 255, 255)
        if right_neck_angle < 150:
            neck_status_color = (14, 14, 232)
            advice_neck = 'Tuck chin in'
    return neck_status_color, advice_neck
