import cv2

def check_webcam_compatibility():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # Change the parameter to use a different webcam if necessary

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    # Read and display video frames
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame from webcam")
            break

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to check webcam compatibility
if __name__ == "__main__":
    check_webcam_compatibility()
