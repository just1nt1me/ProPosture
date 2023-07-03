import cv2
import mediapipe as mp
from proposture.utils import load_video, get_angles, get_landmarks, get_video_dimensions, get_sideview
from proposture.metrics import get_reps_and_stage, get_rep_advice, get_neck, get_hip, get_knee, get_hand, get_hand_align, get_shoulder_elbow_dist
from proposture.visuals import show_status, show_neck, show_hip, show_knee, show_hand, show_align, show_elbow
from proposture.report import get_full_reps, get_pdf

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load video
video_file_path = "media/latest_sideview.mp4"
cap = load_video(video_file_path)
height, width = get_video_dimensions(cap)
advice_list = []

# Set up mediapipe instance
def main(cap, height, width, view = 'front', rep_counter = 0, stage = 'START',
         top_full_rep_counter=0, bottom_full_rep_counter=0, full_rep_stage='start'):
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

            # If sideview, get sideview angle
            if view == 'side':
                sideview_angle = get_sideview(*landmarks)
            if view == 'front':
                sideview_angle = None

            # Calculate angles of joints
            angles = get_angles(*landmarks)

            # Name joint angle variables
            elbow_angles = angles[:2]
            neck_angles = angles[2:4]
            wrist_angles = angles[4:6]
            shoulder_angles = angles[6:8]
            shoulder_distance = angles[12:14]
            shoulder_elbow_distance = angles[14:16]
            hip_angles = angles[8:10]
            knee_angles = angles[10:12]

            # Get status box
            reps_stage = get_reps_and_stage(elbow_angles, rep_counter, stage)
            rep_advice = get_rep_advice(elbow_angles, sideview_angle)
            show_status(image, rep_advice, *reps_stage)

            # Update rep_advice, stage, rep_counter for the next loop
            rep_advice = rep_advice
            stage = reps_stage[0]
            rep_counter = reps_stage[1]

            # Get full reps data for PDF report
            full_reps_data = get_full_reps(elbow_angles, top_full_rep_counter, bottom_full_rep_counter, full_rep_stage)
            top_full_rep_counter = full_reps_data[0]
            bottom_full_rep_counter = full_reps_data[1]
            full_rep_stage = full_reps_data[2]

            if view == 'front':
                align = get_hand_align(shoulder_distance, elbow_angles)
                elbow = get_shoulder_elbow_dist(shoulder_elbow_distance, elbow_angles)

                # Visualize status on joints
                align_status = show_align(image, align, height, width, *landmarks)
                align_status
                if not (align[1] in advice_list):
                    advice_list.append(align[1])

                elbow_status = show_elbow(image, elbow, height, width, *landmarks)
                elbow_status
                if not (elbow[1] in advice_list):
                    advice_list.append(elbow[1])

            # Get status and advice
            if view == 'side':
                neck = get_neck(neck_angles, sideview_angle)
                hand = get_hand(shoulder_distance, wrist_angles, sideview_angle)
                hip = get_hip(hip_angles, sideview_angle)
                knee = get_knee(knee_angles, sideview_angle)

                # Visualize status on joints
                neck_status = show_neck(image, neck, sideview_angle, height, width, *landmarks)
                neck_status
                if not (neck[1] in advice_list):
                    advice_list.append(neck[1])

                hand_status = show_hand(image, hand, sideview_angle, height, width, *landmarks)
                hand_status
                if not (hand[1] in advice_list):
                    advice_list.append(hand[1])

                hip_status = show_hip(image, hip, sideview_angle, height, width, *landmarks)
                hip_status
                if not (hip[1] in advice_list):
                    advice_list.append(hip[1])

                knee_status = show_knee(image, knee, sideview_angle, height, width, *landmarks)
                knee_status
                if not (knee[1] in advice_list):
                    advice_list.append(knee[1])

            # Display advice text
            if len(advice_list) > 0:
                cv2.rectangle(image, (0, 80), (400, 120), (36, 237, 227), -1)
                cv2.putText(image, "POSTURE ADVICE:", (15, 100),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                for i, advice in enumerate(advice_list):
                    rectangle_y = 120 + i * 30  # Adjust the y-coordinate of the rectangle based on the index
                    text_y = 130 + i * 30  # Adjust the y-coordinate of the text based on the index

                    cv2.rectangle(image, (0, rectangle_y), (400, rectangle_y + 30), (36, 237, 227), -1)
                    cv2.putText(image, advice, (15, text_y),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

            # Render detections
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                      mp.solutions.pose.POSE_CONNECTIONS,
                                                      mp.solutions.drawing_utils.DrawingSpec(
                                                          color=(245, 117, 66), thickness=2, circle_radius=2),
                                                      mp.solutions.drawing_utils.DrawingSpec(
                                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                                      )

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Generate frame as JPEG bytes
