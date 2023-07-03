#import libraries
import cv2
import mediapipe as mp
from flask import Flask, render_template, request, Response

from proposture.new_main import main  # Import the main function from your app.py
# Import other necessary modules and functions as needed
from proposture.utils import load_video, get_angles, get_landmarks, get_video_dimensions, get_sideview
from proposture.metrics import get_reps_and_stage, get_rep_advice, get_neck, get_hip, get_knee, get_hand, get_hand_align, get_shoulder_elbow_dist
from proposture.visuals import show_status, show_neck, show_hip, show_knee, show_hand, show_align, show_elbow
from proposture.report import get_full_reps, get_pdf

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

app = Flask(__name__)


'''TODO: main page explains ProPosture
then allow user to select from FRONTVIEW LIVE FEED
- allow access to webcam
- run webcam into model
- show live feed on website
- stop live feed
- generate post performance feedback
or SIDEVIEW VIDEO UPLOAD
- user selects a video to upload
- run video through model'''

@app.route('/')
def index():
    return render_template('index.html')

ALLOWED_EXTENSIONS = ['mp4']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video file found'
    video = request.files['video']
    if video.filename == '':
        return 'No video selected'
    if video and allowed_file(video.filename):
        video_file_path = 'static/videos/' + video.filename
        video.save(video_file_path)
        cap = load_video(video_file_path)
        height, width = get_video_dimensions(cap)
        advice_list = []
        return Response(main(cap, height, width, view='side'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return 'Invalid video file'

@app.route('/livestream')
def livestream():
    cap = cv2.VideoCapture(0)
    height, width = get_video_dimensions(cap)
    advice_list = []
    return Response(main(cap, height, width, view='front'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
