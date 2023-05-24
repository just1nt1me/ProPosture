'''This file takes the angles from data.py and outputs conditions for visuals.py'''

# Pushup position variable
stage = 'START' #default is start > DOWN, UP for reps
quality = None #good, bad for angle
rep_counter = 0 # counts number of reps
min_elbow = None # calculate min elbow angle

#Rep Counter and Up/Down Stage
def get_reps_and_stage(left_elbow_angle, right_elbow_angle):
    if left_elbow_angle and right_elbow_angle < 90:
        stage = "DOWN"
    if left_elbow_angle and right_elbow_angle > 110 and stage =='DOWN':
        stage = "UP"
        rep_counter +=1
    return stage, rep_counter
