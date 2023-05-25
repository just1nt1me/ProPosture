#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

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
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    static_image_mode = args.static_image_mode
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    rev_color = args.rev_color

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # 色指定
    if rev_color:
        color = (255, 255, 255)
        bg_color = (100, 33, 3)
    else:
        color = (100, 33, 3)
        bg_color = (255, 255, 255)

    while True:

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image01 = copy.deepcopy(image)
        debug_image02 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, (0, 0), (image.shape[1], image.shape[0]),
                    bg_color,
                    thickness=-1)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        # 描画 ################################################################
        if results.pose_landmarks is not None:
            # 描画
            debug_image01 = draw_landmarks(
                debug_image01,
                results.pose_landmarks,
            )
            debug_image02 = draw_stick_figure(
                debug_image02,
                results.pose_landmarks,
                color=color,
                bg_color=bg_color,
            )

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('Tokyo2020 Debug', debug_image01)
        cv.imshow('Tokyo2020 Pictogram', debug_image02)

    cap.release()
    cv.destroyAllWindows()


def draw_stick_figure(
        image,
        landmarks,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
        visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    # 各ランドマーク算出
    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append(
            [index, landmark.visibility, (landmark_x, landmark_y), landmark_z])

    # 脚の付け根の位置を腰の中点に修正
    right_leg = landmark_point[23]
    left_leg = landmark_point[24]
    leg_x = int((right_leg[2][0] + left_leg[2][0]) / 2)
    leg_y = int((right_leg[2][1] + left_leg[2][1]) / 2)

    landmark_point[23][2] = (leg_x, leg_y)
    landmark_point[24][2] = (leg_x, leg_y)

    # 距離順にソート
    sorted_landmark_point = sorted(landmark_point,
                                   reverse=True,
                                   key=lambda x: x[3])

    # 各サイズ算出
    (face_x, face_y), face_radius = min_enclosing_face_circle(landmark_point)

    face_x = int(face_x)
    face_y = int(face_y)
    face_radius = int(face_radius * 1.5)

    stick_radius01 = int(face_radius * (4 / 5))
    stick_radius02 = int(stick_radius01 * (3 / 4))
    stick_radius03 = int(stick_radius02 * (3 / 4))

    # 描画対象リスト
    draw_list = [
        11,  # 右腕
        12,  # 左腕
        23,  # 右脚
        24,  # 左脚
    ]

    # 背景色
    cv.rectangle(image, (0, 0), (image_width, image_height),
                 bg_color,
                 thickness=-1)

    # 顔 描画
    cv.circle(image, (face_x, face_y), face_radius, color, -1)

    # 腕/脚 描画
    for landmark_info in sorted_landmark_point:
        index = landmark_info[0]

        if index in draw_list:
            point01 = [p for p in landmark_point if p[0] == index][0]
            point02 = [p for p in landmark_point if p[0] == (index + 2)][0]
            point03 = [p for p in landmark_point if p[0] == (index + 4)][0]

            if point01[1] > visibility_th and point02[1] > visibility_th:
                image = draw_stick(
                    image,
                    point01[2],
                    stick_radius01,
                    point02[2],
                    stick_radius02,
                    color=color,
                    bg_color=bg_color,
                )
            if point02[1] > visibility_th and point03[1] > visibility_th:
                image = draw_stick(
                    image,
                    point02[2],
                    stick_radius02,
                    point03[2],
                    stick_radius03,
                    color=color,
                    bg_color=bg_color,
                )

    return image


def min_enclosing_face_circle(landmark_point):
    landmark_array = np.empty((0, 2), int)

    index_list = [1, 4, 7, 8, 9, 10]
    for index in index_list:
        np_landmark_point = [
            np.array(
                (landmark_point[index][2][0], landmark_point[index][2][1]))
        ]
        landmark_array = np.append(landmark_array, np_landmark_point, axis=0)

    center, radius = cv.minEnclosingCircle(points=landmark_array)

    return center, radius


def draw_stick(
        image,
        point01,
        point01_radius,
        point02,
        point02_radius,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
):
    cv.circle(image, point01, point01_radius, color, -1)
    cv.circle(image, point02, point02_radius, color, -1)

    draw_list = []
    for index in range(2):
        rad = math.atan2(point02[1] - point01[1], point02[0] - point01[0])

        rad = rad + (math.pi / 2) + (math.pi * index)
        point_x = int(point01_radius * math.cos(rad)) + point01[0]
        point_y = int(point01_radius * math.sin(rad)) + point01[1]

        draw_list.append([point_x, point_y])

        point_x = int(point02_radius * math.cos(rad)) + point02[0]
        point_y = int(point02_radius * math.sin(rad)) + point02[1]

        draw_list.append([point_x, point_y])

    points = np.array((draw_list[0], draw_list[1], draw_list[3], draw_list[2]))
    cv.fillConvexPoly(image, points=points, color=color)

    return image


def draw_landmarks(
    image,
    landmarks,
    # upper_body_only,
    visibility_th=0.5,
    show_landmarks=None
):
    image_width, image_height = image.shape[1], image.shape[0]
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        results = pose.process(image)

        try:
                landmarks = results.pose_landmarks.landmark

                # 1. GETTING COORDINATES FOR ALL ANGLES
                ## ALL LEFT BODY PARTS
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                l_ear= [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].z]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]

                ## ALL RIGHT BODY PARTS
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                r_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].z]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]

                # 2. GETTING USEFUL ANGLES ONE-BY-ONE
                ## 2.1 GETTING ELBOW ANGLES --> ALLOW TO CHECK IF FULL REP

                ### LEFT ELBOW
                #### Calculate angle
                left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                l_shoulder_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
                r_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
                avg_shoulder_angle = round((l_shoulder_angle+r_shoulder_angle)/2)
                left_hip_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
                right_hip_angle = calculate_angle(r_shoulder, r_hip, r_ankle)
                new_l_shoulder_angle = calculate_angle(r_shoulder, l_shoulder, l_elbow)
                new_r_shoulder_angle = calculate_angle(l_shoulder, r_shoulder, r_elbow)
                new_avg_shoulder_angle = round((new_l_shoulder_angle+new_r_shoulder_angle)/2)
                wrist_l_shoulder_angle = calculate_angle(r_shoulder, l_shoulder, l_wrist)
                wrist_r_shoulder_angle = calculate_angle(l_shoulder, r_shoulder, r_wrist)
                wrist_avg_shoulder_angle = round((wrist_l_shoulder_angle+wrist_r_shoulder_angle)/2)
                left_hip_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
                right_hip_angle = calculate_angle(r_shoulder, r_hip, r_ankle)
                left_neck_angle = calculate_angle(l_ear,l_shoulder,l_hip)
                right_neck_angle = calculate_angle(r_ear,r_shoulder,r_hip)
                avg_neck_angle = round((left_neck_angle+right_neck_angle)/2)
                left_knee_angle = calculate_angle(l_hip, l_knee,l_ankle)
                right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                avg_knee_angle = round((left_knee_angle+right_knee_angle)/2)
        except:
                pass
        if show_landmarks == 'Show':
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                  mp.solutions.pose.POSE_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 117, 66), thickness=2, circle_radius=2),
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                                  )
            return image
    return image

if __name__ == '__main__':
    main()
