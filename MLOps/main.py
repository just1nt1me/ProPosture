import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import time
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    #radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    #angle = np.abs(radians*180.0/np.pi)

    #Calculate the vectors ba and bc
    ba = a - b
    bc = c - b

    # Calculate the dot product and the magnitudes of the vectors
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Calculate the cosine of the angle using the dot product and magnitudes
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    # Calculate the angle in radians using the arccosine function
    radians = np.arccos(cosine_angle)

    # Convert radians to degrees
    angle = np.degrees(radians)

    if angle >180.0:
        angle = 360-angle

    return round(angle)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    parser.add_argument('--static_image_mode', action='store_true')
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--rev_color', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    static_image_mode = args.static_image_mode
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    rev_color = args.rev_color

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    if rev_color:
        color = (255, 255, 255)
        bg_color = (100, 33, 3)
    else:
        color = (100, 33, 3)
        bg_color = (255, 255, 255)

    while True:

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image01 = copy.deepcopy(image)
        debug_image02 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, (0, 0), (image.shape[1], image.shape[0]),
                    bg_color,
                    thickness=-1)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks is not None:
            debug_image01 = draw_landmarks(
                debug_image01,
                results.pose_landmarks,
            )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('Tokyo2020 Debug', debug_image01)
        cv.imshow('Tokyo2020 Pictogram', debug_image02)

    cap.release()
    cv.destroyAllWindows()


def draw_landmarks(
    image,
    landmarks='Show',
    # upper_body_only,
    visibility_th=0.5,
    video_settings='Show'
):
    image_width, image_height = image.shape[1], image.shape[0]
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        results = pose.process(image)

        try:
                landmarks = results.pose_landmarks.landmark
        except:
                pass
        if video_settings == 'Show':
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                  mp.solutions.pose.POSE_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 117, 66), thickness=2, circle_radius=2),
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                                  )
            return image
        if video_settings == 'Curl Counter':
            counter = 0
            stage = None
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(shoulder, elbow, wrist)
            cv.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA
                                    )
            if angle > 160:
                stage = "down"
            if angle < 30 and stage =='down':
                stage="up"
                counter +=1
                print(counter)
            cv.rectangle(image, (0,0), (225,73), (245,117,16), -1)

            # Rep data
            cv.putText(image, 'REPS', (15,12),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
            cv.putText(image, str(counter),
                        (10,60),
                        cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)

            # Stage data
            cv.putText(image, 'STAGE', (65,12),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
            cv.putText(image, stage,
                        (60,60),
                        cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)


            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            return image
    return image

def load_image(path : str):
        image = tf.io.read_file(path)
        image = tf.compat.v1.image.decode_jpeg(image)
        image = tf.expand_dims(image, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.cast(tf.image.resize_with_pad(image, 160, 256), dtype=tf.int32)
        return image

def preprocess_image(image, new_width, new_height):
    """
    take an frame of a video converted to an image through opencv,
    wth the new_width and new height  for reshaping purpose.
    Based on the image original definition :
    - (480p: 854px by 480px)
    - (720p: 854px by 480px)
    - (1080p: 854px by 480px)
    """
    start = time.time()
    # Resize to the target shape and cast to an int32 vector
    input_image = tf.cast(tf.image.resize_with_pad(image, new_width, new_height), dtype=tf.int32)
    # Create a batch (input tensor)
    # input_image = tf.expand_dims(input_image, axis=0)

    print(input_image.shape)
    return input_image

if __name__ == '__main__':
    main()
