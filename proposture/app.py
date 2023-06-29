from flask import Flask, render_template
from proposture.utils import load_video, get_video_dimensions
from proposture.main import main  # Import the main function from your app.py
# Import other necessary modules and functions as needed

app = Flask(__name__)
# ... your existing code ...

@app.route('/')
def index():
    video_url = "../media/latest_sideview.mp4"  # Update with the actual video file URL
    return render_template('index.html', video_url=video_url)

@app.route('/video')
def video():
    # Call the main function or any other functions from your app.py
    cap = load_video("media/latest_sideview.mp4")
    height, width = get_video_dimensions(cap)
    main(cap, height, width, view='side')
    return "Video processing completed"

if __name__ == "__main__":
    app.run(debug=True)
