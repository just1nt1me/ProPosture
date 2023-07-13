#import libraries
import cv2
from flask import Flask, render_template, request, Response

from proposture.new_main import main  # Import the main function from your app.py
# Import other necessary modules and functions as needed
from proposture.utils import load_video, get_video_dimensions

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
    app.run(debug=False)
