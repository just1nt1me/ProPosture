'''This file takes the angles from data.py and outputs conditions for visuals.py'''

# variables for get_reps_and_stage
# stage = 'START' #default is start > DOWN, UP for reps
# rep_counter = 0 # counts number of reps

#Rep Counter and Up/Down Stage
def get_reps_and_stage(elbow_angles, rep_counter, stage):
    left_elbow_angle, right_elbow_angle = elbow_angles
    if left_elbow_angle and right_elbow_angle < 90:
        stage = "DOWN"
    if left_elbow_angle and right_elbow_angle > 110 and stage =='DOWN':
        stage = "UP"
        rep_counter +=1
    return stage, rep_counter

quality = None #good, bad for angle
