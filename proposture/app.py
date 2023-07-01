from flask import Flask, render_template, request
from proposture.utils import load_video, get_video_dimensions
from proposture.main import main  # Import the main function from your app.py
# Import other necessary modules and functions as needed

app = Flask(__name__)
# ... your existing code ...

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
        video.save('static/videos/' + video.filename)
        return render_template('preview.html', video_name=video.filename)
    return 'Invalid video file'

@app.route('/video')
def video():
    # Call the main function or any other functions from your app.py
    cap = load_video("media/latest_sideview.mp4")
    height, width = get_video_dimensions(cap)
    main(cap, height, width, view='side')
    return "Video processing completed"

if __name__ == "__main__":
    app.run(debug=True)
