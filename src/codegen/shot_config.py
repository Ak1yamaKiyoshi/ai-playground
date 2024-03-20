

class ShotConfig:
    class Coder:
        system_message = """"You are an expert in computer vision and image processing using the OpenCV library. Your task is to provide Python code using the OpenCV library (version 4.9.x) to perform various image processing tasks based on the provided prompts. Your code should be well-documented, efficient, and follow best practices. If any additional information or clarification is needed, kindly request it from the user.""" 
        prompt = """coder prompt"""
        shots = [
            ("""Implement a function that detects and tracks features in a video using the ORB algorithm.""",
             """
import cv2
def detect_and_track_features(video_path):
    '''
    Detects and tracks features in a video using the ORB algorithm.
    Args:
        video_path (str): Path to the video file.
    '''
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize the ORB detector and descriptor
    orb = cv2.ORB_create()

    # Initialize the feature tracker
    tracker = cv2.MultiTracker_create()

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Detect keypoints and compute descriptors
        kp, des = orb.detectAndCompute(frame, None)

        # Track features from the previous frame
        success, boxes = tracker.update(frame)

        # Draw tracked bounding boxes
        for box in boxes:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Feature Tracking', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
"""),
            ("""Create a function that tracks a selected region of interest (ROI) in a video using the Lucas-Kanade algorithm.""", 
             """
             
import cv2
import numpy as np

def track_roi(video_path):
    '''
    Tracks a selected region of interest (ROI) in a video using the Lucas-Kanade algorithm.

    Args:
        video_path (str): Path to the video file.
    '''
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Get the first frame and select the ROI
    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)

    # Initialize the ROI tracker
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Track the ROI points
        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points,
            nextPts=None, **lk_params)

        # Update the ROI points
        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        # Draw the tracked ROI
        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow('ROI Tracking', frame)

        # Update the ROI points for the next iteration
        roi_points = good_new.reshape(-1, 1, 2)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
"""),
        ]
        
    class Namer:
        system_message = """You are an AI assistant tasked with naming code files based on their content. The file names should be concise, descriptive, and no longer than 15 characters."""
        prompt = """Given the code for a program, function, or module, provide a short but descriptive file name for it, adhering to the 15-character limit. The file name should be relevant to the code's functionality and purpose. Use only ASCII characters and avoid special characters or spaces. 
        file shuld start with `run_` and end with `.py`.
        """
        shots = [
            ("""```py
def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a
```""", """run_fib_sequence.py"""),
            ("""function calculateTotal(items) {
  let total = 0;
  for (let item of items) {
    total += item.price * item.quantity;
  }
  return total;
}```""", 
"""run_calc_total.js"""),
        ]
        

    class Debugger:
        system_message = """debugger system_message"""
        prompt = """debugger prompt"""
        shots = [
            ("""debuget shot0 in""", """debuget shot0 out"""),
            ("""debuget shot1 in""", """debuget shot1 out"""),
        ]
    
    class Consoler:
        system_message = """ """
        prompt = """ """
        shots = [
            (""" """, """ """),
            (""" """, """ """),
        ]