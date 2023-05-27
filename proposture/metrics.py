'''This file takes the angles from data.py and outputs conditions for visuals.py'''

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

# NECK conditional statements
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

# TODO: this condition doesn't work well
# HAND conditional statements
def get_hand(shoulder_distance, elbow_angles, sideview_angle, hand_status_color = (31, 194, 53), advice_hand = None):
    left_x_distance, right_x_distance = shoulder_distance
    left_elbow_angle, right_elbow_angle = elbow_angles
    if sideview_angle == "left":
        if left_elbow_angle > 150:
            if left_x_distance < 0.05:
                hand_status_color = (31, 194, 53)
            if left_x_distance > 0.05:
                hand_status_color = (14, 14, 232)
                advice_hand = 'Align hands with shoulders'
    if sideview_angle == "right":
        if right_elbow_angle > 150:
            if right_x_distance < 0.05:
                hand_status_color = (31, 194, 53)
            if right_x_distance > 0.05:
                hand_status_color = (14, 14, 232)
                advice_hand = 'Align hands with shoulders'
    return hand_status_color, advice_hand

# HIP conditional statements
def get_hip(hip_angles, sideview_angle, advice_hip = None):
    left_hip_angle, right_hip_angle = hip_angles
    if sideview_angle == 'left':
        if left_hip_angle >= 170:
            hip_status_color= (31, 194, 53)
        if left_hip_angle in range(160,170):
            hip_status_color= (0, 255, 255)
        if left_hip_angle < 160:
            hip_status_color= (14, 14, 232)
            advice_hip = 'Lower hips'
    if sideview_angle == 'right':
        if right_hip_angle >= 170:
            hip_status_color= (31, 194, 53)
        if right_hip_angle in range(160,170):
            hip_status_color= (0, 255, 255)
        if right_hip_angle < 160:
            hip_status_color = (14, 14, 232)
            advice_hip = 'Lower hips'
    return hip_status_color, advice_hip

# KNEE conditional statements
def get_knee(knee_angles, sideview_angle, advice_knee = None):
    left_knee_angle, right_knee_angle = knee_angles
    if sideview_angle == 'left':
        if left_knee_angle >= 160:
            knee_status_color= (31, 194, 53)
        if left_knee_angle in range(155,160):
            knee_status_color= (0, 255, 255)
        if left_knee_angle < 155:
            knee_status_color= (14, 14, 232)
            advice_knee = 'Straighten legs'
    if sideview_angle == 'right':
        if right_knee_angle >= 160:
            knee_status_color= (31, 194, 53)
        if right_knee_angle in range(155,160):
            knee_status_color= (0, 255, 255)
        if right_knee_angle < 155:
            knee_status_color = (14, 14, 232)
            advice_knee = 'Straighten legs'
    return knee_status_color, advice_knee
