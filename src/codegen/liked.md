### AI
#### 20.03.2024 | 18:36:35
#### Filename: detect_track_obj.py
#### Summary: Object detection and tracking in videos using OpenCV, with background subtraction for detection and optical flow for tracking, displaying rectangles around detected objects and lines for object movement.
#### Code: 
```py
import cv2
import numpy as np
import traceback

def detect_object(video_path):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_object(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_pts, _, _ = cv2.calcOpticalFlowPyrLK(prev_frame, frame_gray, prev_pts, None, **lk_params)

        for new, old in zip(new_pts, prev_pts):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            cv2.line(frame, (x_new, y_new), (x_old, y_old), (0, 255, 0), 2)
            cv2.circle(frame, (x_new, y_new), 5, (0, 255, 0), -1)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame_gray.copy()
        prev_pts = new_pts.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object(video_path)
    track_object(video_path)

if __name__ == "__main__":
    import os
    from typing import List

    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            track(video)
        except:
            traceback.print_exc()

```
