import copy
from multiprocessing import Queue, Process
from typing import NamedTuple, List

import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, ClientSettings

import av
import cv2 as cv
import numpy as np
import mediapipe as mp

from main import draw_landmarks, draw_stick_figure

from fake_objects import FakeResultObject, FakeLandmarksObject, FakeLandmarkObject

from turn import get_ice_servers


_SENTINEL_ = "_SENTINEL_"


def pose_process(
    in_queue: Queue,
    out_queue: Queue
):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    while True:
        input_item = in_queue.get(timeout=10)
        if isinstance(input_item, type(_SENTINEL_)) and input_item == _SENTINEL_:
            break

        results = pose.process(input_item)
        picklable_results = FakeResultObject(pose_landmarks=FakeLandmarksObject(landmark=[
            FakeLandmarkObject(
                x=pose_landmark.x,
                y=pose_landmark.y,
                z=pose_landmark.z,
                visibility=pose_landmark.visibility,
            ) for pose_landmark in results.pose_landmarks.landmark
        ]))
        out_queue.put_nowait(picklable_results)


class Tokyo2020PictogramVideoProcessor(VideoProcessorBase):
    def __init__(self, show_landmarks=None) -> None:
        self._in_queue = Queue()
        self._out_queue = Queue()
        self.show_landmarks=show_landmarks,
        self._pose_process = Process(target=pose_process, kwargs={
            "in_queue": self._in_queue,
            "out_queue": self._out_queue,
        })

        self._pose_process.start()

    def _infer_pose(self, image):
        self._in_queue.put_nowait(image)
        return self._out_queue.get(timeout=10)

    def _stop_pose_process(self):
        self._in_queue.put_nowait(_SENTINEL_)
        self._pose_process.join(timeout=10)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        # カメラキャプチャ #####################################################
        image = frame.to_ndarray(format="bgr24")

        image = cv.flip(image, 1)  # ミラー表示
        debug_image01 = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self._infer_pose(image)
        # results = self._pose.process(image)
        # 描画 ################################################################
        if results.pose_landmarks is not None:
            # 描画
            print(f'from recv: {self.show_landmarks}')
            debug_image01 = draw_landmarks(
                debug_image01,
                results.pose_landmarks,
                show_landmarks=self.show_landmarks
            )
        return av.VideoFrame.from_ndarray(debug_image01, format="bgr24")

    def __del__(self):
        print("Stop the inference process...")
        self._stop_pose_process()
        print("Stopped!")


def main():
    with st.expander("If you want to film yourself from the front"):

        show_landmarks = st.radio("Landmarks", ['None', 'Show'])

        def processor_factory():
            return Tokyo2020PictogramVideoProcessor(show_landmarks=show_landmarks)

        webrtc_ctx = webrtc_streamer(
            key="tokyo2020-Pictogram",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=processor_factory,
        )
        st.session_state["started"] = webrtc_ctx.state.playing

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.show_landmarks = show_landmarks

    with st.expander("If you want to film yourself from the front"):

        show_landmarks = st.radio("Landmarks",['None', 'Show'])

        def processor_factory():
            return Tokyo2020PictogramVideoProcessor(show_landmarks=show_landmarks)

        webrtc_ctx = webrtc_streamer(
            key="tokyo2020-Pictogram",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=processor_factory,
        )
        st.session_state["started"] = webrtc_ctx.state.playing

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.show_landmarks = show_landmarks


if __name__ == "__main__":
    main()
