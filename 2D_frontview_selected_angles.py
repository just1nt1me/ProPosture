import cv2
import mediapipe as mp
import numpy as np
import os
import math
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#function to calculate angles
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = round(np.abs(radians*180.0/np.pi))

    if angle >180.0:
        angle = 360-angle

    return angle

path=os.path.join(os.path.dirname(os.getcwd()),"ProPosture","media","frontview_pushupbadgood.mp4")


# CAREFUL: THIS FILE OPERATES ON A LIVEFEED

cap = cv2.VideoCapture(path)

# Pushup position variable
stage = None
full_rep_stage = None
quality = None
counter = 0
top_full_rep_counter = 0
bottom_full_rep_counter = 0
elbow_angle_list=[]

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
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # 1. GETTING COORDINATES FOR ALL ANGLES
            ## ALL LEFT BODY PARTS
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            l_ear= [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            ## ALL RIGHT BODY PARTS
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            r_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # 2. GETTING USEFUL ANGLES ONE-BY-ONE

            ## 2.1 GETTING ELBOW ANGLES --> ALLOW TO CHECK IF FULL REP
            ### Compute angles
            left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            ### Compute average angle
            average_elbow_angle=(left_elbow_angle+right_elbow_angle)/2
            ### Visualise
            average_elbow_text_position = (image.shape[1] - 220, 30)
            cv2.putText(image, str(f'Elbows angle: {average_elbow_angle}'),
                           average_elbow_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            elbow_angle_list.append(average_elbow_angle)


            ##2.2 COMPUTE X-DISTANCE BETWEEN SHOULDER AND WRIST
            ### Compute distances
            left_x_distance = round(abs(l_wrist[0]-l_shoulder[0]),4)
            right_x_distance = round(abs(r_wrist[0]-r_shoulder[0]),4)
            ### Compute average distance
            x_distances_mean=round(abs((left_x_distance+right_x_distance)/2),4)
            ### Visualise
            x_distances_mean_text_position = (image.shape[1] - 220, 70)
            cv2.putText(image, str(f'Wrist-shoulder: {x_distances_mean}'),
                           x_distances_mean_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##2.3 COMPUTE X-DISTANCE BETWEEN BOTH WRISTS/SHOULDERS AND BOTH ELBOWS
            ### Compute distances
            shoulders_x_distance = round(abs(l_shoulder[0]-r_shoulder[0]),3)
            elbows_x_distance = round(abs(l_elbow[0]-r_elbow[0]),3)
            ### Compute ratio of distances
            shoulder_elbow_ratio=round(elbows_x_distance/shoulders_x_distance,3)
            ### Visualise
            shoulder_elbow_ratio_text_position = (image.shape[1] - 220, 110)
            cv2.putText(image, str(f'Wrist-shoulder: {shoulder_elbow_ratio}'),
                           shoulder_elbow_ratio_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            #3 QUALITY LOGIC TESTING

            ##3.1 REP COUNTER
            if average_elbow_angle > 160 and stage =='down':
                counter +=1
            if average_elbow_angle > 160:
                stage = "up"
            if average_elbow_angle < 90:
                stage="down"
            rep_counter_text_position = (30, 30)
            # rep_stage_text_position = (30, 60)
            cv2.putText(image, str(f'Reps count: {counter}'),
                        rep_counter_text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            # cv2.putText(image, str(f'Rep stage: {stage}'),
            #             rep_stage_text_position,
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if average_elbow_angle > 175 and full_rep_stage =='down':
                top_full_rep_counter +=1
            if average_elbow_angle > 175:
                full_rep_stage = "up"
            if average_elbow_angle < 60 and full_rep_stage == 'up':
                bottom_full_rep_counter +=1
            if average_elbow_angle < 60:
                full_rep_stage="down"

            # full_rep_counter_text_position = (30, 50)
            # cv2.putText(image, str(f'Full rep counter: {top_full_rep_counter}'),
            #             full_rep_counter_text_position,
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            # btm_full_rep_counter_text_position = (30, 70)
            # cv2.putText(image, str(f'Full rep counter2: {bottom_full_rep_counter}'),
            #             btm_full_rep_counter_text_position,
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            ##3.2 FULL REP VERIFIER
            if average_elbow_angle in range(170,180):
                advice = "GOOD JOB U LOSER ! Now down"
            if average_elbow_angle in range(0,65):
                advice="GOOD JOB U LOSER ! Now push up"
            rep_advice_text_position = (30, 70)
            cv2.putText(image, str(f'{advice}'),
                        rep_advice_text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            ##3.3 HAND-SHOULDER X-AXIS ALIGNMENT
            if average_elbow_angle>150:
                if x_distances_mean<=0.075:
                    hands_status="Good"
                if x_distances_mean>0.075:
                    hands_status="Bad"
                    advice_hands = 'Align hands with shoulder'
            hands_status_text_position = (30,110)
            cv2.putText(image, str(f'Hands position: {hands_status}'),
                        hands_status_text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            ##3.4 SHOULDERS-ELBOWS RATIO
            if average_elbow_angle<75:
                if shoulder_elbow_ratio<2:
                    elbow_status="Good"
                if shoulder_elbow_ratio>2:
                    elbow_status="Bad"
                    advice_elbows = "Tuck elbows in"
            elbows_status_text_position = (30, 150)
            cv2.putText(image, str(f'Elbows position: {elbow_status}'),
                        elbows_status_text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        except:
            pass

        #4 CREATING LABELS AND STUFF

        #4.1 Setup status box
        # cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        # #4.2 Status data
        # cv2.putText(image, 'Quality', (15,12),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # cv2.putText(image, quality,
        #             (10,60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # !! WE KEEP THIS PART TO MAYBE LATER DISPLAY NUMBER OF REPS
        #cv2.putText(image, 'STAGE', (65,12),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        #cv2.putText(image, stage,
        #            (60,60),
        #            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        #4.3 Render detections
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

top_rep_performance = 100*top_full_rep_counter/counter
bottom_rep_performance = 100*(bottom_full_rep_counter+1)/counter

#Generating a PDF Report

import subprocess
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

# Sample metrics
repetitions = counter
perfect_execution_percentage_top = top_rep_performance
perfect_execution_percentage_bottom = bottom_rep_performance
hands_position_score = 85.2

# Create a PDF document
pdf = SimpleDocTemplate("performance_review.pdf", pagesize=letter)

# Define table data
data = [
    ['Metrics', 'Score'],
    ['Repetitions', repetitions],
    ['Proportion of pushups perfect at the top', '{}%'.format(perfect_execution_percentage_top)],
    ['Proportion of pushups perfect at the bottom', '{}%'.format(perfect_execution_percentage_bottom)],
    ['Hands Position', '{}%'.format(hands_position_score)],
]

# Define table style
table_style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 14),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
])

# Conditionally apply style to "perfect execution" value cell
if perfect_execution_percentage_top > 85.0:
    table_style.add('BACKGROUND', (1, 2), (1, -1), colors.green)

# Create the table and apply style
table = Table(data)
table.setStyle(table_style)

# Build the table and add it to the PDF document
elements = []
elements.append(table)
pdf.build(elements)


# Open the generated PDF with the default PDF viewer
subprocess.Popen(["open", "performance_review.pdf"])
