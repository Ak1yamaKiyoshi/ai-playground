# video_tracking.py
### Date: 20.03.2024-13:51:35
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)

        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
            pass

```
# opticalflow.py
### Date: 20.03.2024-14:09:20
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points,
            nextPts=None, **lk_params)

        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
            pass

```
# track_video.py
### Date: 20.03.2024-14:09:57
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)
        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
            pass

```
# `optical_track.py`
### Date: 20.03.2024-14:12:19
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)
        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import os
from typing import List

if __name__ == "__main__":
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
            pass

```
# video_track.py
### Date: 20.03.2024-14:25:17
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)
        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
            pass

```
# video_track.py
### Date: 20.03.2024-14:26:08
### Meta: opencv-gen-test
### Purpose: coder
```py
import os
import cv2
import numpy as np
from typing import List

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points,
            nextPts=None, **lk_params)

        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
            pass

```
# optical_flow.py
### Date: 20.03.2024-14:30:51
### Meta: opencv-gen-test
### Purpose: coder
```py
import os
import cv2
import numpy as np
from typing import List

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)

    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)

        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
            pass

```
# optical_track.py
### Date: 20.03.2024-14:32:23
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)

        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
            pass

```
# py
track_roi_lk.py

### Date: 20.03.2024-14:32:45
### Meta: opencv-gen-test
### Purpose: coder
```py
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

    # Check if the video capture is open
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Initialize the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Select the ROI in the first frame
        if 'roi_points' not in locals():
            roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
            roi_mask = np.zeros_like(frame[:, :, 0])
            roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
            roi_points = np.column_stack((np.where(roi_mask == 1)))

        # Track the ROI points using Lucas-Kanade
        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)

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

if __name__ == "__main__":
    video_path = "path_to_your_video.mp4"  # Specify the path to your video file
    track_roi(video_path)

```
# optical_flow.py
### Date: 20.03.2024-14:33:47
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)
        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
            pass

```
# video_roi_track
### Date: 20.03.2024-14:35:04
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)
        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
            roi_mask = np.zeros_like(frame[:, :, 0])
            roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
            roi_points = np.column_stack((np.where(roi_mask == 1)))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track("path_to_your_video.mp4")

```
# py
track_roi.py

### Date: 20.03.2024-14:35:51
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if 'roi' not in locals():
            roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
            roi_mask = np.zeros_like(frame[:, :, 0])
            roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
            roi_points = np.column_stack((np.where(roi_mask == 1)))

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)
        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
            roi_mask = np.zeros_like(frame[:, :, 0])
            roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
            roi_points = np.column_stack((np.where(roi_mask == 1)))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track("path_to_your_video.mp4")

```
# py
run_track.py

### Date: 20.03.2024-14:39:44
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def track(video_path):
    '''
    Tracks an object in a video using the Lucas-Kanade algorithm.

    Args:
        video_path (str): Path to the video file.
    '''
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Get the first frame and select the object to track
    ret, frame = cap.read()
    bbox = cv2.selectROI('Select Object to Track', frame, fromCenter=False)

    # Initialize the object tracker
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, bbox)

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Update the object tracker
        success, bbox = tracker.update(frame)

        # Draw the tracked object
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Object Tracking', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            track(video)
        except Exception as e:
            print(f"Error tracking object in {video}: {e}")

```
# run_track_obj.py
### Date: 20.03.2024-14:43:53
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def track(video_path):
    '''
    Tracks objects in a video using the Lucas-Kanade algorithm.

    Args:
        video_path (str): Path to the video file.
    '''
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Get the first frame
    ret, frame = cap.read()

    # Select the object to track
    bbox = cv2.selectROI('Select Object to Track', frame, fromCenter=False)
    x, y, w, h = bbox

    # Initialize the object tracker
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(roi_gray)
    mask[0:h, 0:w] = 255

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, frame_gray, (x, y, w, h), None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = np.array([[x, y]])

        # Update the object position
        if len(good_new) > 0:
            x, y = good_new.ravel()

        # Draw the object rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Object Tracking', frame)

        # Update the ROI for the next iteration
        roi = frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            track(video)
        except Exception as e:
            print(f"Error processing video: {video}. {e}")

```
# run_track_obj.py
### Date: 20.03.2024-14:44:54
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def track(video_path):
    '''
    Tracks objects in a video using the Lucas-Kanade algorithm.

    Args:
        video_path (str): Path to the video file.
    '''
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if 'old_frame_gray' not in locals():
            old_frame_gray = frame_gray
            continue

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray, frame_gray, None, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = np.array([[x, y] for x, y in p1[st == 1]])

        # Draw the object rectangles
        for x, y in good_old:
            cv2.rectangle(frame, (x-5, y-5), (x+5, y+5), (0, 255, 0), 2)

        cv2.imshow('Object Tracking', frame)

        # Update the old frame
        old_frame_gray = frame_gray.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            track(video)
        except Exception as e:
            print(f"Error processing video: {video}. {e}")

```
# run_track_video.py
### Date: 20.03.2024-14:46:28
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def track(video_path):
    '''
    Tracks objects in a video using the Lucas-Kanade algorithm.

    Args:
        video_path (str): Path to the video file.
    '''
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if 'old_frame_gray' not in locals():
            old_frame_gray = frame_gray
            continue

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray, frame_gray, None, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = np.array([[x, y] for x, y in p1[st == 1]])

        # Draw the object rectangles
        for x, y in good_old:
            cv2.rectangle(frame, (x-5, y-5), (x+5, y+5), (0, 255, 0), 2)

        cv2.imshow('Object Tracking', frame)

        # Update the old frame
        old_frame_gray = frame_gray.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            track(video)
        except Exception as e:
            print(f"Error processing video: {video}. {e}")

```
# run_track_obj.py
### Date: 20.03.2024-14:57:12
### Meta: opencv-gen-test
### Purpose: coder
```py
import os
from typing import List
import cv2
import numpy as np

def track(video_path):
    '''
    Tracks objects in a video using the Lucas-Kanade algorithm.

    Args:
        video_path (str): Path to the video file.
    '''
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Get the first frame
    ret, frame = cap.read()
    if not ret:
        return

    # Select the object to track
    bbox = cv2.selectROI('Select Object to Track', frame, fromCenter=False)
    x, y, w, h = bbox

    # Initialize the object tracker
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame)
    mask[y:y+h, x:x+w] = 255

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Lucas-Kanade
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, frame_gray, (x, y, w, h), None, **lk_params)

        if p1 is not None:
            x, y, w, h = p1.ravel()

            # Draw bounding box around the tracked object
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

            cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            track(video)
        except Exception as e:
            print(f"Error processing video: {video}, {e}")

```
# run_track_obj.py
### Date: 20.03.2024-14:58:02
### Meta: opencv-gen-test
### Purpose: coder
```py
import os
from typing import List
import cv2
import numpy as np

def track(video_path):
    '''
    Tracks objects in a video using the Lucas-Kanade algorithm.

    Args:
        video_path (str): Path to the video file.
    '''
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Get the first frame
    ret, frame = cap.read()
    if not ret:
        return

    # Initialize the object tracker
    bbox = (0, 0, frame.shape[1], frame.shape[0])  # Initialize with full frame
    x, y, w, h = bbox
    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame)
    mask[y:y+h, x:x+w] = 255

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Lucas-Kanade
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, frame_gray, (x, y, w, h), None, **lk_params)

        if p1 is not None:
            x, y, w, h = p1.ravel()

            # Draw bounding box around the tracked object
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

            cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            track(video)
        except Exception as e:
            print(f"Error processing video: {video}, {e}")

```
# run_track_roi.py
### Date: 20.03.2024-14:58:32
### Meta: opencv-gen-test
### Purpose: coder
```py
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
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Get the first frame and select the ROI
    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)

    # Initialize the ROI tracker
    x, y, w, h = roi
    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_points = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track the ROI points
        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev=roi_gray, next=frame_gray, prevPts=roi_points,
            nextPts=None, **lk_params)

        # Update the ROI points
        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        if len(good_new) > 0:
            x, y = good_new.ravel()

            # Draw a rectangle around the tracked ROI
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('ROI Tracking', frame)

        # Update the ROI points for the next iteration
        roi_points = good_new.reshape(-1, 1, 2)
        roi_gray = frame_gray.copy()

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = './assets/IMG_2084.mp4'  # Update with your video path
    track_roi(video_path)

```
# run_auto_roi.py
### Date: 20.03.2024-15:02:33
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def automatic_roi_detection(frame):
    # Implement your automatic ROI detection logic here
    # This can be done using background subtraction, object detection, or any other suitable method
    # For simplicity, we will use a fixed ROI in this example
    roi = (100, 100, 50, 50)  # Example ROI (x, y, width, height)
    return roi

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    roi = None
    roi_points = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if roi is None:
            roi = automatic_roi_detection(frame)
            roi_mask = np.zeros_like(frame[:, :, 0])
            roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
            roi_points = np.column_stack((np.where(roi_mask == 1)))

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)

        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if len(good_new) == 0:
            roi = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
            pass

```
# run_auto_roi.py
### Date: 20.03.2024-15:03:58
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def automatic_roi_detection(frame):
    # Example automatic ROI detection using a fixed ROI
    roi = (100, 100, 50, 50)  # Example ROI (x, y, width, height)
    return roi

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    roi = None
    roi_points = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if roi is None:
            roi = automatic_roi_detection(frame)
            roi_mask = np.zeros_like(frame[:, :, 0])
            roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
            roi_points = np.column_stack((np.where(roi_mask == 1)))

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)

        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if len(good_new) == 0:
            roi = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
            pass

```
# run_auto_roi.py
### Date: 20.03.2024-15:04:59
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List

def automatic_roi_detection(frame):
    # Example automatic ROI detection using a fixed ROI
    roi = (100, 100, 50, 50)  # Example ROI (x, y, width, height)
    return roi

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    roi = None
    roi_points = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if roi is None:
            roi = automatic_roi_detection(frame)
            roi_mask = np.zeros_like(frame[:, :, 0])
            roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
            roi_points = np.column_stack((np.where(roi_mask == 1)))

        new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points, nextPts=None, **lk_params)

        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if len(good_new) == 0:
            roi = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
            pass

```
# run_track_obj.py
### Date: 20.03.2024-15:06:07
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def track(video_path):
    '''
    Tracks objects in a video using the Lucas-Kanade algorithm.
    Automatically detects and tracks ROIs in the video.
    
    Args:
        video_path (str): Path to the video file.
    '''
    cap = cv2.VideoCapture(video_path)
    
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    roi_mask = None
    roi_points = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if roi_mask is None:
            roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
            roi_mask = np.zeros_like(frame[:, :, 0])
            roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
            roi_points = np.column_stack((np.where(roi_mask == 1)))
        else:
            new_roi_points, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=frame, nextImg=frame, prevPts=roi_points, nextPts=None, **lk_params)
            good_new = new_roi_points[status == 1]
            good_old = roi_points[status == 1]
            
            for pt in good_new.astype(int):
                cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)
            
            cv2.imshow('Object Tracking', frame)
            roi_points = good_new.reshape(-1, 1, 2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            roi_mask = None
            roi_points = None
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback
    
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
# run_auto_roi.py
### Date: 20.03.2024-15:07:22
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def track(video_path):
    '''
    Tracks objects in a video using the Lucas-Kanade algorithm.
    Automatically detects and tracks ROIs in the video.
    
    Args:
        video_path (str): Path to the video file.
    '''
    cap = cv2.VideoCapture(video_path)
    
    lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    roi_mask = None
    roi_points = None
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if roi_mask is None:
            roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
            roi_mask = np.zeros_like(frame[:, :, 0])
            roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
            roi_points = np.column_stack((np.where(roi_mask == 1)))
            kalman.statePost = np.array([[roi[0], roi[1], 0, 0]], np.float32).T
        else:
            prediction = kalman.predict()
            new_roi_points, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=frame, nextImg=frame, prevPts=roi_points, nextPts=None, **lk_params)
            good_new = new_roi_points[status == 1]
            good_old = roi_points[status == 1]
            
            for pt in good_new.astype(int):
                cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)
            
            cv2.imshow('Object Tracking', frame)
            roi_points = good_new.reshape(-1, 1, 2)
            
            measurement = roi_points[0].reshape(2, 1)
            kalman.correct(measurement)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            roi_mask = None
            roi_points = None
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback
    
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
# run_obj_detect.py
### Date: 20.03.2024-15:10:30
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    roi_mask = None
    roi_points = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > 500:  # Set a threshold for object size
                    roi_mask = np.zeros_like(frame[:, :, 0])
                    roi_mask[y:y+h, x:x+w] = 1
                    roi_points = np.column_stack((np.where(roi_mask == 1)))
                    kalman.statePost = np.array([[x, y, 0, 0]], np.float32).T
                    break

        if roi_mask is not None:
            prediction = kalman.predict()
            new_roi_points, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=frame, nextImg=frame, prevPts=roi_points, nextPts=None, winSize=(15, 15))

            good_new = new_roi_points[status == 1]
            good_old = roi_points[status == 1]

            for pt in good_new.astype(int):
                cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

            cv2.imshow('Object Tracking', frame)
            roi_points = good_new.reshape(-1, 1, 2)

            measurement = roi_points[0].reshape(2, 1)
            kalman.correct(measurement)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            roi_mask = None
            roi_points = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)

    for video in videos:
        try:
            detect_objects(video)
        except:
            traceback.print_exc()

```
# run_track_roi.py
### Date: 20.03.2024-15:12:20
### Meta: opencv-gen-test
### Purpose: coder
```py
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
        new_roi_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=frame, nextImg=frame, prevPts=roi_points.reshape(-1, 1, 2).astype(np.float32),
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

if __name__ == "__main__":
    track_roi('path_to_video_file.mp4')

```
# run_track_roi.py
### Date: 20.03.2024-15:13:14
### Meta: opencv-gen-test
### Purpose: coder
```py
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

    # Get the first frame
    ret, frame = cap.read()

    # Select ROI
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
    cv2.destroyWindow('Select ROI')

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
        new_roi_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=frame, nextImg=frame, prevPts=roi_points.reshape(-1, 1, 2).astype(np.float32),
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

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
            cv2.destroyWindow('Select ROI')
            roi_mask = np.zeros_like(frame[:, :, 0])
            roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
            roi_points = np.column_stack((np.where(roi_mask == 1)))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_roi('path_to_video_file.mp4')

```
# run_track_roi.py
### Date: 20.03.2024-15:14:14
### Meta: opencv-gen-test
### Purpose: coder
```py
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

    # Initialize ROI variables
    roi = None
    roi_selected = False

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        if not roi_selected:
            cv2.imshow('Select ROI', frame)
        else:
            cv2.imshow('ROI Tracking', frame)

        key = cv2.waitKey(1) & 0xFF

        # Check for key press
        if key == ord('q'):
            break
        elif key == ord('c'):
            roi = None
            roi_selected = False
        elif key == ord(' ') or key == 13:  # SPACE or ENTER key
            if roi is None:
                roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
                cv2.destroyWindow('Select ROI')
                roi_selected = True

        if roi_selected:
            # Initialize the ROI tracker
            roi_mask = np.zeros_like(frame[:, :, 0])
            roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
            roi_points = np.column_stack((np.where(roi_mask == 1)))

            # Track the ROI points
            new_roi_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prevImg=frame, nextImg=frame, prevPts=roi_points.reshape(-1, 1, 2).astype(np.float32),
                nextPts=None, **lk_params)

            # Update the ROI points
            good_new = new_roi_points[status == 1]
            good_old = roi_points[status == 1]

            # Draw the tracked ROI
            for pt in good_new.astype(int):
                cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_roi('path_to_video_file.mp4')

```
# run_obj_track.py
### Date: 20.03.2024-15:23:29
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List

def detect_object(frame, bg_subtractor, kalman_filter):
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the detected ROI list
    detected_rois = []

    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        roi = (x, y, w, h)

        # Predict the next state with Kalman Filter
        prediction = kalman_filter.predict()

        # Update the measurement based on the detected ROI
        measurement = np.array([[x + w/2], [y + h/2]], dtype=np.float32)
        kalman_filter.correct(measurement)

        detected_rois.append(roi)

    return detected_rois

def track_roi(frame, roi):
    # Initialize the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Initialize the ROI tracker
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    # Track the ROI points
    new_roi_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev=frame, next=frame, prevPts=roi_points,
        nextPts=None, **lk_params)

    # Update the ROI points
    good_new = new_roi_points[status == 1]
    good_old = roi_points[status == 1]

    return good_new

def track(video_path):
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize the background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Initialize the Kalman Filter
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                [0, 1, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32)
    kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32) * 0.03

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Stage 1: Object Detection with Background Subtractor and Kalman Filter
        detected_rois = detect_object(frame, bg_subtractor, kalman_filter)

        # Stage 2: Track the detected ROI using Lucas-Kanade
        for roi in detected_rois:
            roi_points = track_roi(frame, roi)

            # Draw the tracked ROI
            for pt in roi_points.astype(int):
                cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow('Object Tracking', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# run_track_obj.py
### Date: 20.03.2024-15:26:45
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List

def detect_object(frame, bg_subtractor, kalman_filter):
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize bounding boxes list
    bounding_boxes = []

    for contour in contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    if len(bounding_boxes) > 0:
        # Select the first bounding box for tracking
        x, y, w, h = bounding_boxes[0]
        kalman_filter.init((x + w//2, y + h//2), (0, 0))

    return bounding_boxes

def track(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Initialize Kalman filter
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stage 1: Object Detection
        bounding_boxes = detect_object(frame, bg_subtractor, kalman_filter)

        for box in bounding_boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Stage 2: Object Tracking with Lucas-Kanade
        if len(bounding_boxes) > 0:
            roi = frame[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_points = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

            if roi_points is not None:
                roi_points += np.array([x, y]).reshape(1, 2)
                roi_points = roi_points.reshape(-1, 1, 2)

                new_roi_points, _, _ = cv2.calcOpticalFlowPyrLK(frame, frame, roi_points, None, winSize=(15, 15), maxLevel=2)

                for pt in new_roi_points.astype(int):
                    cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    import traceback
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
# run_obj_track.py
### Date: 20.03.2024-15:34:16
### Meta: opencv-gen-test
### Purpose: coder
```py
import os
from typing import List
import cv2
import numpy as np

def detect_object(frame, bg_subtractor, kalman_filter):
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Apply morphological operations to remove noise
    fg_mask = cv2.erode(fg_mask, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize ROI list
    rois = []

    for contour in contours:
        # Compute bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Predict the next state using Kalman filter
        prediction = kalman_filter.predict()

        # Update the measurement
        measurement = np.array([[x + w/2], [y + h/2]], dtype=np.float32)
        kalman_filter.correct(measurement)

        # Add ROI to the list
        rois.append((x, y, w, h))

    return rois

def track_roi(frame, roi):
    # Initialize the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Initialize the ROI tracker
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    # Track the ROI points
    new_roi_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev=frame, next=frame, prevPts=roi_points,
        nextPts=None, **lk_params)

    # Update the ROI points
    good_new = new_roi_points[status == 1]
    good_old = roi_points[status == 1]

    # Draw the tracked ROI
    for pt in good_new.astype(int):
        cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

    return good_new, frame

def track(video_path):
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Initialize Kalman filter
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    kalman_filter.processNoiseCov = 0.01 * np.eye(4, dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if 'rois' not in locals():
            rois = detect_object(frame, bg_subtractor, kalman_filter)

        if len(rois) == 0:
            del rois
            continue

        for roi in rois:
            roi_points, frame = track_roi(frame, roi)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
# run_track_roi.py
### Date: 20.03.2024-15:36:15
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def track_roi(frame, roi):
    '''
    Tracks a selected region of interest (ROI) in a frame using the Lucas-Kanade algorithm.

    Args:
        frame (numpy.ndarray): Input frame.
        roi (tuple): Region of interest (x, y, w, h).
    
    Returns:
        numpy.ndarray: Updated frame with tracked ROI.
    '''
    # Initialize the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Extract ROI coordinates
    x, y, w, h = roi

    # Select the ROI region
    roi_frame = frame[y:y+h, x:x+w]

    # Convert the ROI to grayscale
    roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Initialize the ROI points
    roi_points = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track the ROI points using Lucas-Kanade
        new_roi_points, status, _ = cv2.calcOpticalFlowPyrLK(roi_gray, frame_gray, roi_points, None, **lk_params)

        # Update the ROI points
        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        # Update the ROI coordinates
        x_new, y_new = good_new[0]
        x_diff, y_diff = x_new - x, y_new - y
        x += x_diff
        y += y_diff

        # Draw the tracked ROI
        cv2.rectangle(frame, (int(x), int(y)), (int(x) + w, int(y) + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('ROI Tracking', frame)

        # Update the ROI points for the next iteration
        roi_points = good_new.reshape(-1, 1, 2)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return frame


```
# run_detect_track.py
### Date: 20.03.2024-15:39:46
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List

def detect_object(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
    kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32) * 0.03
    kalman_filter.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 0.1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center = (x + w // 2, y + h // 2)
                kalman_filter.correct(np.array([[np.float32(center[0])], [np.float32(center[1])]]))

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)

    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_roi_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points,
            nextPts=None, **lk_params)

        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if len(roi_points) == 0:
            detect_object(video_path)
            track_roi(video_path)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object(video_path)
    track_roi(video_path)

import os
from typing import List
import traceback

if __name__ == "__main__":
    videos:List[str] = []
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
# run_obj_track.py
### Date: 20.03.2024-15:42:30
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
    kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32) * 0.03
    kalman_filter.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 0.1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center = (x + w // 2, y + h // 2)
                kalman_filter.correct(np.array([[np.float32(center[0])], [np.float32(center[1])]]))

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center = (x + w // 2, y + h // 2)
                kalman_filter.correct(np.array([[np.float32(center[0])], [np.float32(center[1])]]))

        roi_mask = np.zeros_like(frame[:, :, 0])
        roi_mask[int(y):int(y + h), int(x):int(x + w)] = 1
        roi_points = np.column_stack((np.where(roi_mask == 1)))

        new_roi_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points,
            nextPts=None, **lk_params)

        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if len(roi_points) == 0:
            detect_object(video_path)
            track_roi(video_path)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object(video_path)
    track_roi(video_path)

import os
from typing import List
import traceback

if __name__ == "__main__":
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
# run_track_roi.py
### Date: 20.03.2024-15:44:55
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def track_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()

    if not ret:
        print("Error reading video.")
        return

    roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
    roi_mask = np.zeros_like(frame[:, :, 0])
    roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 1
    roi_points = np.column_stack((np.where(roi_mask == 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = cv2.createBackgroundSubtractorMOG2().apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center = (x + w // 2, y + h // 2)

        new_roi_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev=frame, next=frame, prevPts=roi_points,
            nextPts=None, **lk_params)

        good_new = new_roi_points[status == 1]
        good_old = roi_points[status == 1]

        for pt in good_new.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        cv2.imshow('ROI Tracking', frame)

        roi_points = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import os
from typing import List
import traceback

if __name__ == "__main__":
    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            track_roi(video)
        except:
            traceback.print_exc()

```
# run_track_vid.py
### Date: 20.03.2024-15:46:57
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List
import traceback

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()

    if not ret:
        print("Error reading video.")
        return

    fg_mask = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = fg_mask.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center = (x + w // 2, y + h // 2)

        cv2.imshow('ROI Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
# run_track_roi.py
### Date: 20.03.2024-15:48:01
### Meta: opencv-gen-test
### Purpose: coder
```py
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

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            track_roi(video)
        except:
            traceback.print_exc()

```
# run_obj_track.py
### Date: 20.03.2024-15:50:13
### Meta: opencv-gen-test
### Purpose: coder
```py
import os
from typing import List
import cv2
import numpy as np

def detect_object(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 1000:
            bbox.append((x, y, w, h))
    
    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if 'bbox' not in locals():
            bbox = detect_object(frame, bg_subtractor)
            if len(bbox) == 0:
                continue
            x, y, w, h = bbox[0]
            kalman.statePost = np.array([[x + w/2], [y + h/2], [0], [0]], np.float32)
            kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        prediction = kalman.predict()
        x, y, w, h = int(prediction[0] - w/2), int(prediction[1] - h/2), w, h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        bbox = detect_object(frame, bg_subtractor)
        if len(bbox) == 0:
            continue
        x, y, w, h = bbox[0]
        measurement = np.array([[x + w/2], [y + h/2]], np.float32)
        
        kalman.correct(measurement)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
# run_track_obj.py
### Date: 20.03.2024-15:51:30
### Meta: opencv-gen-test
### Purpose: coder
```py
import os
from typing import List
import cv2
import numpy as np
import traceback

def detect_object(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 1000:
            bbox.append((x, y, w, h))
    
    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if 'bbox' not in locals():
            bbox = detect_object(frame, bg_subtractor)
            if len(bbox) == 0:
                continue
            x, y, w, h = bbox[0]
            kalman.statePost = np.array([[x + w/2], [y + h/2], [0], [0]], np.float32)
            kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        prediction = kalman.predict()
        x, y, w, h = int(prediction[0] - w/2), int(prediction[1] - h/2), w, h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        bbox = detect_object(frame, bg_subtractor)
        if len(bbox) == 0:
            continue
        x, y, w, h = bbox[0]
        measurement = np.array([[x + w/2], [y + h/2]], np.float32)
        
        kalman.correct(measurement)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
# run_obj_track.py
### Date: 20.03.2024-15:52:28
### Meta: opencv-gen-test
### Purpose: coder
```py
import os
from typing import List
import cv2
import numpy as np
import traceback

def detect_object(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 1000:
            bbox.append((x, y, w, h))
    
    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if 'bbox' not in locals():
            bbox = detect_object(frame, bg_subtractor)
            if len(bbox) == 0:
                continue
            x, y, w, h = bbox[0]
            kalman.statePost = np.array([[x + w/2], [y + h/2], [0], [0]], np.float32)
            kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        prediction = kalman.predict()
        x, y, w, h = int(prediction[0] - w/2), int(prediction[1] - h/2), w, h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        bbox = detect_object(frame, bg_subtractor)
        if len(bbox) == 0:
            continue
        x, y, w, h = bbox[0]
        measurement = np.array([[x + w/2], [y + h/2]], np.float32)
        
        kalman.correct(measurement)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
# run_obj_track.py
### Date: 20.03.2024-15:52:57
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 1000:
            bbox.append((x, y, w, h))
    
    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if 'bbox' not in locals():
            bbox = detect_object(frame, bg_subtractor)
            if len(bbox) == 0:
                continue
            x, y, w, h = bbox[0]
            kalman.statePost = np.array([[x + w/2], [y + h/2], [0], [0]], np.float32)
            kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        prediction = kalman.predict()
        x, y, w, h = int(prediction[0] - w/2), int(prediction[1] - h/2), w, h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        bbox = detect_object(frame, bg_subtractor)
        if len(bbox) == 0:
            continue
        x, y, w, h = bbox[0]
        measurement = np.array([[x + w/2], [y + h/2]], np.float32)
        
        kalman.correct(measurement)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# run_obj_track.py
### Date: 20.03.2024-15:56:06
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 1000:
            bbox.append((x, y, w, h))
    
    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if 'bbox' not in locals():
            bbox = detect_object(frame, bg_subtractor)
            if len(bbox) == 0:
                continue
            x, y, w, h = bbox[0]
            kalman.statePost = np.array([[x + w/2], [y + h/2], [0], [0]], np.float32)
            kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        prediction = kalman.predict()
        x, y, w, h = int(prediction[0] - w/2), int(prediction[1] - h/2), w, h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        bbox = detect_object(frame, bg_subtractor)
        if len(bbox) == 0:
            continue
        x, y, w, h = bbox[0]
        measurement = np.array([[x + w/2], [y + h/2]], np.float32)
        
        kalman.correct(measurement)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# run_track_obj.py
### Date: 20.03.2024-15:59:41
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 1000:
            bbox.append((x, y, w, h))
    
    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if 'bbox' not in locals():
            bbox = detect_object(frame, bg_subtractor)
            if len(bbox) == 0:
                continue
            x, y, w, h = bbox[0]
            kalman.statePost = np.array([[x + w/2], [y + h/2], [0], [0]], np.float32)
            kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        prediction = kalman.predict()
        x, y, w, h = int(prediction[0] - w/2), int(prediction[1] - h/2), w, h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        bbox = detect_object(frame, bg_subtractor)
        if len(bbox) == 0:
            continue
        x, y, w, h = bbox[0]
        measurement = np.array([[x + w/2], [y + h/2]], np.float32)
        
        kalman.correct(measurement)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# object_track.py
### Date: 20.03.2024-16:43:13
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List

def detect_object(frame, bg_subtractor, kalman_filter):
    '''
    Detects objects using background subtraction and Kalman filter.
    Args:
        frame (numpy.ndarray): Input frame.
        bg_subtractor: Background subtractor object.
        kalman_filter: Kalman filter object.
    Returns:
        bbox (tuple): Bounding box coordinates (x, y, w, h).
    '''
    fg_mask = bg_subtractor.apply(frame)
    fg_mask[fg_mask < 200] = 0

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(max_contour)
    bbox = (x, y, w, h)

    # Kalman filter prediction
    prediction = kalman_filter.predict()
    kalman_filter.correct((x + w / 2, y + h / 2))

    return bbox

def track_object(frame, bbox):
    '''
    Tracks the object using Lucas-Kanade algorithm.
    Args:
        frame (numpy.ndarray): Input frame.
        bbox (tuple): Bounding box coordinates (x, y, w, h).
    Returns:
        bbox (tuple): Updated bounding box coordinates (x, y, w, h).
    '''
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, frame, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) > 0:
            dx, dy = np.mean(good_new - good_old, axis=0)
            x += int(dx)
            y += int(dy)

    return x, y, w, h

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
    kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32) * 0.03
    kalman_filter.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 0.03
    kalman_filter.errorCovPost = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if 'bbox' not in locals():
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        else:
            bbox = track_object(frame, bbox)

        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# object_track_upd.py
### Date: 20.03.2024-16:49:15
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List

def detect_object(frame, bg_subtractor, kalman_filter):
    '''
    Detects objects using background subtraction and Kalman filter.
    Args:
        frame (numpy.ndarray): Input frame.
        bg_subtractor: Background subtractor object.
        kalman_filter: Kalman filter object.
    Returns:
        bbox (tuple): Bounding box coordinates (x, y, w, h).
    '''
    fg_mask = bg_subtractor.apply(frame)
    fg_mask[fg_mask < 200] = 0

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(max_contour)
    bbox = (x, y, w, h)

    # Kalman filter prediction
    measurement = np.array([[x + w / 2], [y + h / 2]], dtype=np.float32)
    kalman_filter.correct(measurement)

    return bbox

def track_object(frame, bbox):
    '''
    Tracks the object using Lucas-Kanade algorithm.
    Args:
        frame (numpy.ndarray): Input frame.
        bbox (tuple): Bounding box coordinates (x, y, w, h).
    Returns:
        bbox (tuple): Updated bounding box coordinates (x, y, w, h).
    '''
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, frame, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) > 0:
            dx, dy = np.mean(good_new - good_old, axis=0)
            x += int(dx)
            y += int(dy)

    return x, y, w, h

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
    kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32) * 0.03
    kalman_filter.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 0.03
    kalman_filter.errorCovPost = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if 'bbox' not in locals():
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        else:
            bbox = track_object(frame, bbox)

        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# object_track_upd2.py
### Date: 20.03.2024-16:51:08
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List

def detect_object(frame, bg_subtractor, kalman_filter):
    '''
    Detects objects using background subtraction and Kalman filter.
    Args:
        frame (numpy.ndarray): Input frame.
        bg_subtractor: Background subtractor object.
        kalman_filter: Kalman filter object.
    Returns:
        bbox (tuple): Bounding box coordinates (x, y, w, h).
    '''
    fg_mask = bg_subtractor.apply(frame)
    fg_mask[fg_mask < 200] = 0

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        bbox = (x, y, w, h)

        # Kalman filter prediction
        measurement = np.array([[x + w / 2], [y + h / 2]], dtype=np.float32)
        kalman_filter.correct(measurement)

        return bbox
    else:
        return None

def track_object(frame, bbox):
    '''
    Tracks the object using Lucas-Kanade algorithm.
    Args:
        frame (numpy.ndarray): Input frame.
        bbox (tuple): Bounding box coordinates (x, y, w, h).
    Returns:
        bbox (tuple): Updated bounding box coordinates (x, y, w, h).
    '''
    if bbox:
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, frame, p0, None, **lk_params)

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 0:
                dx, dy = np.mean(good_new - good_old, axis=0)
                x += int(dx)
                y += int(dy)

        return x, y, w, h
    else:
        return None

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
    kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32) * 0.03
    kalman_filter.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 0.03
    kalman_filter.errorCovPost = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if 'bbox' not in locals():
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        else:
            bbox = track_object(frame, bbox)

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# object_track_upd3.py
### Date: 20.03.2024-16:54:10
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List

def detect_object(frame, bg_subtractor, kalman_filter):
    '''
    Detects objects using background subtraction and Kalman filter.
    Args:
        frame (numpy.ndarray): Input frame.
        bg_subtractor: Background subtractor object.
        kalman_filter: Kalman filter object.
    Returns:
        bbox (tuple): Bounding box coordinates (x, y, w, h).
    '''
    fg_mask = bg_subtractor.apply(frame)
    fg_mask[fg_mask < 200] = 0

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        bbox = (x, y, w, h)

        # Kalman filter prediction
        measurement = np.array([[x + w / 2], [y + h / 2]], dtype=np.float32)
        kalman_filter.correct(measurement)

        return bbox
    else:
        return None

def track_object(frame, bbox):
    '''
    Tracks the object using Lucas-Kanade algorithm.
    Args:
        frame (numpy.ndarray): Input frame.
        bbox (tuple): Bounding box coordinates (x, y, w, h).
    Returns:
        bbox (tuple): Updated bounding box coordinates (x, y, w, h).
    '''
    if bbox:
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, frame, p0, None, **lk_params)

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 0:
                dx, dy = np.mean(good_new - good_old, axis=0)
                x += int(dx)
                y += int(dy)

        return x, y, w, h
    else:
        return None

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
    kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32) * 0.03
    kalman_filter.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 0.03
    kalman_filter.errorCovPost = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if 'bbox' not in locals():
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        else:
            bbox = track_object(frame, bbox)

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# detect_track_medianflow.py
### Date: 20.03.2024-16:54:57
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
from typing import List

def detect_and_track_object(video_path):
    '''
    Detects and tracks object using MedianFlow tracker.
    Args:
        video_path (str): Path to the video file.
    '''
    cap = cv2.VideoCapture(video_path)
    
    tracker = cv2.TrackerMedianFlow_create()

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error reading video file")
        return

    # Select ROI
    bbox = cv2.selectROI("Select Object", frame)
    
    # Initialize tracker with first frame and bounding box
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracker
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            detect_and_track_object(video)
        except:
            traceback.print_exc()

```
# detect_track_boosting.py
### Date: 20.03.2024-16:55:45
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
from typing import List

def detect_and_track_object(video_path):
    '''
    Detects and tracks object using BOOSTING tracker.
    Args:
        video_path (str): Path to the video file.
    '''
    cap = cv2.VideoCapture(video_path)
    
    tracker = cv2.TrackerBoosting_create()

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error reading video file")
        return

    # Select ROI
    bbox = cv2.selectROI("Select Object", frame)
    
    # Initialize tracker with first frame and bounding box
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracker
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            detect_and_track_object(video)
        except:
            traceback.print_exc()

```
# detect_track_mil.py
### Date: 20.03.2024-16:57:19
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
from typing import List

def detect_and_track_object(video_path):
    '''
    Detects and tracks object using MIL tracker.
    Args:
        video_path (str): Path to the video file.
    '''
    cap = cv2.VideoCapture(video_path)
    
    tracker = cv2.TrackerMIL_create()

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error reading video file")
        return

    # Select ROI
    bbox = cv2.selectROI("Select Object", frame)
    
    # Initialize tracker with first frame and bounding box
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracker
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            detect_and_track_object(video)
        except:
            traceback.print_exc()

```
# detect_track_goturn.py
### Date: 20.03.2024-17:00:38
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
from typing import List

def detect_and_track_object(video_path):
    '''
    Detects and tracks object using GOTURN tracker.
    Args:
        video_path (str): Path to the video file.
    '''
    tracker = cv2.TrackerGOTURN_create()

    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error reading video file")
        return

    # Select ROI
    bbox = cv2.selectROI("Select Object", frame)
    
    # Initialize tracker with first frame and bounding box
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracker
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

    videos: List[str] = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            video_path = os.path.join("./assets", path)
            videos.append(video_path)

    videos = sorted(videos)
    for video in videos:
        try:
            detect_and_track_object(video)
        except:
            traceback.print_exc()

```
# It seems like the code snippet was cut off. Would you like me to provide a complete version of the code for object tracking using the KCF tracker?
### Date: 20.03.2024-17:01:09
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
from typing import List

def detect_and_track_object(video_path):
    '''
    Detects and tracks object using KCF tracker.
    Args:
        video_path (str): Path to the video file.
    '''
    tracker = cv2.TrackerKCF_create()

    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error reading video file")
        return

    # Select ROI
    bbox = cv2.selectROI("Select Object", frame)
    
    # Initialize tracker with first frame and bounding box
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracker
        success, bbox =
```
# track_objects.py
### Date: 20.03.2024-18:07:00
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List

def detect_object(frame, bg_subtractor, kalman_filter):
    '''
    Detects objects in a frame using background subtraction and Kalman filter.
    Args:
        frame (numpy.ndarray): Input frame.
        bg_subtractor: Background subtractor object.
        kalman_filter: Kalman filter object.
    Returns:
        List: List of bounding boxes of detected objects.
    '''
    fg_mask = bg_subtractor.apply(frame)
    fg_mask[fg_mask < 200] = 0

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = (x, y, w, h)
            bboxes.append(bbox)

    return bboxes

def track_object(frame, bbox, lk_params):
    '''
    Tracks the object using Lucas-Kanade algorithm.
    Args:
        frame (numpy.ndarray): Input frame.
        bbox (tuple): Bounding box coordinates.
        lk_params: Lucas-Kanade parameters.
    Returns:
        tuple: Updated bounding box coordinates.
    '''
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **lk_params)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, frame, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) > 0:
            dx = int(np.mean(good_new[:, 0]) - np.mean(good_old[:, 0]))
            dy = int(np.mean(good_new[:, 1]) - np.mean(good_old[:, 1]))
            x += dx
            y += dy

    return x, y, w, h

def track(video_path):
    '''
    Tracks objects in a video using background subtraction and Lucas-Kanade algorithm.
    Args:
        video_path (str): Path to the video file.
    '''
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

    bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if bbox is None:
            bboxes = detect_object(frame, bg_subtractor, kalman)
            if len(bboxes) > 0:
                bbox = bboxes[0]
                kalman.statePost = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
            else:
                continue

        bbox = track_object(frame, bbox, lk_params)

        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# track_object_detect.py
### Date: 20.03.2024-18:10:36
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List
import os
import traceback

def detect_object(frame, bg_subtractor, kalman_filter):
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Apply morphological operations to remove noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox = None

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, w, h)
            break

    if bbox:
        # Initialize the Kalman filter with the detected bounding box
        kalman_filter.init(bbox)

    return bbox

def track_object(frame, bbox, lk_params):
    # Convert bbox to points
    bbox_points = np.array([[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]], dtype=np.float32)

    # Track the bbox points using Lucas-Kanade optical flow
    new_bbox_points, _, _ = cv2.calcOpticalFlowPyrLK(prevImg=frame, nextImg=frame, prevPts=bbox_points, nextPts=None, **lk_params)

    # Update the bbox
    bbox = (new_bbox_points[0, 0], new_bbox_points[0, 1], bbox[2], bbox[3])

    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Initialize Kalman filter
    kalman_filter = cv2.TrackerKCF_create()

    # Initialize Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    bbox = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if bbox is None:
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        else:
            bbox = track_object(frame, bbox, lk_params)

        if bbox is None:
            bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            kalman_filter = cv2.TrackerKCF_create()
            continue

        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
# track_obj_detect.py
### Date: 20.03.2024-18:11:58
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List
import os
import traceback

def detect_object(frame, bg_subtractor, kalman_filter):
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Apply morphological operations to remove noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox = None

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, w, h)
            break

    if bbox:
        # Initialize the Kalman filter with the detected bounding box
        kalman_filter.init(bbox)

    return bbox

def track_object(frame, bbox, lk_params):
    # Convert bbox to points
    bbox_points = np.array([[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]], dtype=np.float32)

    # Track the bbox points using Lucas-Kanade optical flow
    new_bbox_points, _, _ = cv2.calcOpticalFlowPyrLK(prevImg=frame, nextImg=frame, prevPts=bbox_points, nextPts=None, **lk_params)

    # Update the bbox
    bbox = (new_bbox_points[0, 0], new_bbox_points[0, 1], bbox[2], bbox[3])

    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Initialize Kalman filter
    kalman_filter = cv2.TrackerKCF_create()

    # Initialize Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    bbox = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if bbox is None:
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        else:
            bbox = track_object(frame, bbox, lk_params)

        if bbox is None:
            bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            kalman_filter = cv2.TrackerKCF_create()
            continue

        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
# track_object.py
### Date: 20.03.2024-18:17:52
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object(frame, bg_subtractor, kalman_filter):
    fg_mask = bg_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, w, h)
            kalman_filter.correct(np.array([(x + w/2, y + h/2)], dtype=np.float32))
            return bbox
    
    return None

def track_bbox(frame, bbox, lk_params):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    if p0 is not None:
        p0 = np.array([[[x + pt[0][0], y + pt[0][1]]] for pt in p0], dtype=np.float32)
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame, roi, p0, None, **lk_params)
        
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        dx = np.mean(good_new[:, 0, 0] - good_old[:, 0, 0])
        dy = np.mean(good_new[:, 0, 1] - good_old[:, 0, 1])
        
        return (x + dx, y + dy, w, h)
    
    return None

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    bbox = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if bbox is None:
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        else:
            bbox = track_bbox(frame, bbox, lk_params)
        
        if bbox is None:
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# track_object.py
### Date: 20.03.2024-18:19:41
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object(frame, bg_subtractor, kalman_filter):
    fg_mask = bg_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, w, h)
            kalman_filter.correct(np.array([(x + w/2, y + h/2)], dtype=np.float32))
            return bbox
    
    return None

def track_bbox(frame, bbox, lk_params):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    if p0 is not None:
        p0 = np.array([[[x + pt[0][0], y + pt[0][1]]] for pt in p0], dtype=np.float32)
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame, roi, p0, None, **lk_params)
        
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        dx = np.mean(good_new[:, 0, 0] - good_old[:, 0, 0])
        dy = np.mean(good_new[:, 0, 1] - good_old[:, 0, 1])
        
        return (x + dx, y + dy, w, h)
    
    return None

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    bbox = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if bbox is None:
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        else:
            bbox = track_bbox(frame, bbox, lk_params)
        
        if bbox is None:
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# track_object.py
### Date: 20.03.2024-18:20:15
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object(frame, bg_subtractor, kalman_filter):
    fg_mask = bg_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, w, h)
            kalman_filter.correct(np.array([(x + w/2, y + h/2)], dtype=np.float32))
            return bbox
    
    return None

def track_bbox(frame, bbox, lk_params):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    if p0 is not None:
        p0 = np.array([[[x + pt[0][0], y + pt[0][1]]] for pt in p0], dtype=np.float32)
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame, roi, p0, None, **lk_params)
        
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        dx = np.mean(good_new[:, 0, 0] - good_old[:, 0, 0])
        dy = np.mean(good_new[:, 0, 1] - good_old[:, 0, 1])
        
        return (x + dx, y + dy, w, h)
    
    return None

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    bbox = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if bbox is None:
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        else:
            bbox = track_bbox(frame, bbox, lk_params)
        
        if bbox is None:
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# track_object.py
### Date: 20.03.2024-18:22:14
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object(frame, bg_subtractor, kalman_filter):
    fg_mask = bg_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = None
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        bbox = (x, y, w, h)
        
        kalman_filter.correct(np.array([[x + w/2], [y + h/2]], dtype=np.float32))
    
    return bbox

def track_bbox(frame, bbox, lk_params):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **lk_params)
    p0 = np.array([[[x + pt[0][0], y + pt[0][1]]] for pt in p0], dtype=np.float32)
    
    p1, _, _ = cv2.calcOpticalFlowPyrLK(roi_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)
    
    p0 = p0.reshape(-1, 2)
    p1 = p1.reshape(-1, 2)
    
    return p0, p1

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    bbox = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if bbox is None:
            bbox = detect_object(frame, bg_subtractor, kalman_filter)
        else:
            p0, p1 = track_bbox(frame, bbox, lk_params)
            
            if len(p1) == 0:
                bbox = None
            else:
                x, y, w, h = cv2.boundingRect(p1)
                bbox = (x, y, w, h)
        
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    from typing import List
    import traceback
    
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
# track_object.py
### Date: 20.03.2024-18:23:27
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List

def detect_object(frame, background_subtractor, kalman_filter):
    fg_mask = background_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = (x, y, w, h)
            kalman_filter.correct(np.array([(x + w/2), (y + h/2)], dtype=np.float32))
            return bbox
    
    return None

def track_bbox(frame, bbox, lk_params):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    if p0 is not None:
        p0 = np.array(p0).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(roi_gray, cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) > 0:
            x_new, y_new = np.mean(good_new, axis=0).astype(int).ravel()
            bbox = (x + x_new, y + y_new, w, h)
    
    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], dtype=np.float32)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], dtype=np.float32)
    kalman_filter.processNoiseCov = np.eye(4) * 0.03
    kalman_filter.measurementNoiseCov = np.eye(2) * 0.1
    kalman_filter.errorCovPost = np.eye(4)
    
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    bbox = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if bbox is None:
            bbox = detect_object(frame, background_subtractor, kalman_filter)
        else:
            bbox = track_bbox(frame, bbox, lk_params)
        
        if bbox is None:
            background_subtractor = cv2.createBackgroundSubtractorMOG2()
            kalman_filter = cv2.KalmanFilter(4, 2)
            kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]], dtype=np.float32)
            kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0]], dtype=np.float32)
            kalman_filter.processNoiseCov = np.eye(4) * 0.03
            kalman_filter.measurementNoiseCov = np.eye(2) * 0.1
            kalman_filter.errorCovPost = np.eye(4)
        
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
# track_object.py
### Date: 20.03.2024-18:24:33
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List
import traceback

def detect_object(frame, background_subtractor, kalman_filter):
    fg_mask = background_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = (x, y, w, h)
            kalman_filter.correct(np.array([(x + w/2), (y + h/2)], dtype=np.float32))
            return bbox
    
    return None

def track_bbox(frame, bbox, lk_params):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    if p0 is not None:
        p0 = np.array(p0).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(roi_gray, cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) > 0:
            x_new, y_new = np.mean(good_new, axis=0).astype(int).ravel()
            bbox = (x + x_new, y + y_new, w, h)
    
    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], dtype=np.float32)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], dtype=np.float32)
    kalman_filter.processNoiseCov = np.eye(4) * 0.03
    kalman_filter.measurementNoiseCov = np.eye(2) * 0.1
    kalman_filter.errorCovPost = np.eye(4)
    
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    bbox = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if bbox is None:
            bbox = detect_object(frame, background_subtractor, kalman_filter)
        else:
            bbox = track_bbox(frame, bbox, lk_params)
        
        if bbox is None:
            background_subtractor = cv2.createBackgroundSubtractorMOG2()
            kalman_filter = cv2.KalmanFilter(4, 2)
            kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]], dtype=np.float32)
            kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0]], dtype=np.float32)
            kalman_filter.processNoiseCov = np.eye(4) * 0.03
            kalman_filter.measurementNoiseCov = np.eye(2) * 0.1
            kalman_filter.errorCovPost = np.eye(4)
        
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
# track_object.py
### Date: 20.03.2024-18:25:09
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List
import traceback

def detect_object(frame, background_subtractor, kalman_filter):
    fg_mask = background_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = (x, y, w, h)
            kalman_filter.correct(np.array([(x + w/2), (y + h/2)], dtype=np.float32))
            return bbox
    
    return None

def track_bbox(frame, bbox, lk_params):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    if p0 is not None:
        p0 = np.array(p0).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(roi_gray, cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) > 0:
            x_new, y_new = np.mean(good_new, axis=0).astype(int).ravel()
            bbox = (x + x_new, y + y_new, w, h)
    
    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], dtype=np.float32)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], dtype=np.float32)
    kalman_filter.processNoiseCov = np.eye(4) * 0.03
    kalman_filter.measurementNoiseCov = np.eye(2) * 0.1
    kalman_filter.errorCovPost = np.eye(4)
    
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    bbox = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if bbox is None:
            bbox = detect_object(frame, background_subtractor, kalman_filter)
        else:
            bbox = track_bbox(frame, bbox, lk_params)
        
        if bbox is None:
            background_subtractor = cv2.createBackgroundSubtractorMOG2()
            kalman_filter = cv2.KalmanFilter(4, 2)
            kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]], dtype=np.float32)
            kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0]], dtype=np.float32)
            kalman_filter.processNoiseCov = np.eye(4) * 0.03
            kalman_filter.measurementNoiseCov = np.eye(2) * 0.1
            kalman_filter.errorCovPost = np.eye(4)
        
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
# track_object.py
### Date: 20.03.2024-18:25:46
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import os
from typing import List
import traceback

def detect_object(frame, background_subtractor, kalman_filter):
    fg_mask = background_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = (x, y, w, h)
            kalman_filter.correct(np.array([(x + w/2), (y + h/2)], dtype=np.float32))
            return bbox
    
    return None

def track_bbox(frame, bbox, lk_params):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    if p0 is not None:
        p0 = np.array(p0).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(roi_gray, cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) > 0:
            x_new, y_new = np.mean(good_new, axis=0).astype(int).ravel()
            bbox = (x + x_new, y + y_new, w, h)
    
    return bbox

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], dtype=np.float32)
    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], dtype=np.float32)
    kalman_filter.processNoiseCov = np.eye(4) * 0.03
    kalman_filter.measurementNoiseCov = np.eye(2) * 0.1
    kalman_filter.errorCovPost = np.eye(4)
    
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    bbox = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if bbox is None:
            bbox = detect_object(frame, background_subtractor, kalman_filter)
        else:
            bbox = track_bbox(frame, bbox, lk_params)
        
        if bbox is None:
            background_subtractor = cv2.createBackgroundSubtractorMOG2()
            kalman_filter = cv2.KalmanFilter(4, 2)
            kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]], dtype=np.float32)
            kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0]], dtype=np.float32)
            kalman_filter.processNoiseCov = np.eye(4) * 0.03
            kalman_filter.measurementNoiseCov = np.eye(2) * 0.1
            kalman_filter.errorCovPost = np.eye(4)
        
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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
# obj_detect_track.py
### Date: 20.03.2024-18:28:06
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2, 0)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                kalman_filter.correct(np.array([[x + w/2], [y + h/2]], np.float32))
                prediction = kalman_filter.predict()
                bbox = (prediction[0]-w/2, prediction[1]-h/2, w, h)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 2)
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def track_bbox(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ret, frame = cap.read()
    bbox = cv2.selectROI('Select Object to Track', frame, fromCenter=False)
    bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    tracker = cv2.TrackerKCF_create()
    init_bb = (bbox[0], bbox[1], bbox[2], bbox[3])
    ok = tracker.init(frame, init_bb)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ok, bbox = tracker.update(frame)
        if ok:
            bbox = tuple(map(int, bbox))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
        else:
            break
        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    while True:
        detect_objects(video_path)
        track_bbox(video_path)

import os
from typing import List
import traceback

if __name__ == "__main__":
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
# obj_detect_track.py
### Date: 20.03.2024-18:29:33
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2, 0)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                kalman_filter.correct(np.array([[x + w/2], [y + h/2]], np.float32))
                prediction = kalman_filter.predict()
                bbox = (prediction[0]-w/2, prediction[1]-h/2, w, h)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 2)
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return bbox

def track_bbox(video_path, bbox):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ret, frame = cap.read()
    tracker = cv2.TrackerKCF_create()
    ok = tracker.init(frame, bbox)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ok, bbox = tracker.update(frame)
        if ok:
            bbox = tuple(map(int, bbox))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
        else:
            break
        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return bbox

def track(video_path):
    bbox = detect_objects(video_path)
    while True:
        bbox = track_bbox(video_path, bbox)
        if bbox == (0, 0, 0, 0):
            bbox = detect_objects(video_path)

import os
from typing import List
import traceback

if __name__ == "__main__":
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
# obj_detect_track.py
### Date: 20.03.2024-18:30:19
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return bbox

def track_bbox(video_path, bbox):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ret, frame = cap.read()
    bbox = cv2.selectROI('Select Object to Track', frame, fromCenter=False)
    bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    tracker = cv2.TrackerBoosting_create()
    ok = tracker.init(frame, bbox)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ok, bbox = tracker.update(frame)
        if ok:
            bbox = tuple(map(int, bbox))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
        else:
            bbox = (0, 0, 0, 0)
            break
        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return bbox

def track(video_path):
    bbox = detect_objects(video_path)
    while True:
        bbox = track_bbox(video_path, bbox)
        if bbox == (0, 0, 0, 0):
            bbox = detect_objects(video_path)

import os
from typing import List
import traceback

if __name__ == "__main__":
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
# detect_track_obj.py
### Date: 20.03.2024-18:34:02
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
from typing import List

def detect_object(video_path):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    measurement = np.zeros((2, 1), np.float32)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            measurement[0, 0] = x + w/2
            measurement[1, 0] = y + h/2
            kalman.correct(measurement)
            prediction = kalman.predict()
            bbox = (int(prediction[0])-w//2, int(prediction[1])-h//2, w, h)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_bbox(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ret, frame = cap.read()
    bbox = cv2.selectROI('Select ROI', frame, fromCenter=False)
    bbox = np.array([bbox], dtype=np.float32)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        new_bbox, _, _ = cv2.calcOpticalFlowPyrLK(prevImg=frame, nextImg=frame, prevPts=bbox, nextPts=None, **lk_params)
        good_new = new_bbox[0]
        bbox[0] = good_new
        x, y, w, h = bbox[0]
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object(video_path)
    track_bbox(video_path)

if __name__ == "__main__":
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
# detect_track_obj.py
### Date: 20.03.2024-18:35:09
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np
import traceback

def detect_object(video_path):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    measurement = np.zeros((2, 1), np.float32)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            measurement[0, 0] = x + w/2
            measurement[1, 0] = y + h/2
            kalman.correct(measurement)
            prediction = kalman.predict()
            bbox = (int(prediction[0])-w//2, int(prediction[1])-h//2, w, h)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_bbox(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ret, frame = cap.read()
    bbox = cv2.selectROI('Select ROI', frame, fromCenter=False)
    bbox = np.array([bbox], dtype=np.float32)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        new_bbox, _, _ = cv2.calcOpticalFlowPyrLK(prevImg=frame, nextImg=frame, prevPts=bbox, nextPts=None, **lk_params)
        good_new = new_bbox[0]
        bbox[0] = good_new
        x, y, w, h = bbox[0]
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object(video_path)
    track_bbox(video_path)

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
# detect_track_obj.py
### Date: 20.03.2024-18:36:12
### Meta: opencv-gen-test
### Purpose: coder
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
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_bbox(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    bbox = cv2.selectROI('Select ROI', frame, fromCenter=False)
    bbox = np.array([bbox], dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_bbox, _, _ = cv2.calcOpticalFlowPyrLK(prevImg=frame, nextImg=frame, prevPts=bbox, nextPts=None, **lk_params)
        good_new = new_bbox[0]
        bbox[0] = good_new

        x, y, w, h = bbox[0]
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object(video_path)
    track_bbox(video_path)

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
# detect_track_obj.py
### Date: 20.03.2024-18:36:53
### Meta: opencv-gen-test
### Purpose: coder
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
# detect_track_obj.py
### Date: 20.03.2024-18:38:30
### Meta: opencv-gen-test
### Purpose: coder
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
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
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
# track_obj_detect.py
### Date: 20.03.2024-18:40:02
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def track(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Background subtraction
    back_sub = cv2.createBackgroundSubtractorMOG2()

    # Optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Object detection and tracking variables
    object_detected = False
    bbox = None
    tracker = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fg_mask = back_sub.apply(frame)

        if not object_detected:
            # Object detection using contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:  # Adjust the area threshold as needed
                    x, y, w, h = cv2.boundingRect(cnt)
                    bbox = (x, y, w, h)
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, bbox)
                    object_detected = True
                    break

        if object_detected:
            # Update the tracker
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    videos = []
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
# track_obj_detect.py
### Date: 20.03.2024-18:46:08
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object_with_background_subtractor(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], np.float32) * 0.03

    bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, w, h)
                kalman.correct(np.array([[x + w / 2], [y + h / 2]], np.float32))

        prediction = kalman.predict()
        cv2.rectangle(frame, (int(prediction[0] - bbox[2] / 2), int(prediction[1] - bbox[3] / 2),
                              bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_bounding_box_with_lucas_kanade(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply Lucas-Kanade optical flow to track the bounding box
        # Implement Lucas-Kanade tracking here

        cv2.imshow('Lucas-Kanade Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object_with_background_subtractor(video_path)
    while True:
        track_bounding_box_with_lucas_kanade(video_path)
        if bbox_count == 0:
            detect_object_with_background_subtractor(video_path)

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# track_obj_detect.py
### Date: 20.03.2024-18:47:07
### Meta: opencv-gen-test
### Purpose: coder
```py
py
import cv2
import numpy as np

def detect_object_with_background_subtractor(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], np.float32) * 0.03

    bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, w, h)
                kalman.correct(np.array([[x + w / 2], [y + h / 2]], np.float32))

        prediction = kalman.predict()
        cv2.rectangle(frame, (int(prediction[0] - bbox[2] / 2), int(prediction[1] - bbox[3] / 2),
                              bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_bounding_box_with_lucas_kanade(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply Lucas-Kanade optical flow to track the bounding box
        # Implement Lucas-Kanade tracking here

        cv2.imshow('Lucas-Kanade Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object_with_background_subtractor(video_path)
    while True:
        track_bounding_box_with_lucas_kanade(video_path)
        if bbox_count == 0:
            detect_object_with_background_subtractor(video_path)

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# track_obj_detect.py
### Date: 20.03.2024-18:48:57
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object_with_background_subtractor(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], np.float32) * 0.03

    bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, w, h)
                kalman.correct(np.array([[x + w / 2], [y + h / 2]], np.float32))

        prediction = kalman.predict()
        cv2.rectangle(frame, (int(prediction[0] - bbox[2] / 2), int(prediction[1] - bbox[3] / 2),
                              bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_bounding_box_with_lucas_kanade(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Implement Lucas-Kanade optical flow to track the bounding box
        # Implement Lucas-Kanade tracking here

        cv2.imshow('Lucas-Kanade Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object_with_background_subtractor(video_path)
    while True:
        track_bounding_box_with_lucas_kanade(video_path)
        if bbox_count == 0:
            detect_object_with_background_subtractor(video_path)

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# track_obj_track.py
### Date: 20.03.2024-18:49:48
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object_with_background_subtractor(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], np.float32) * 0.03

    bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, w, h)
                kalman.correct(np.array([[x + w / 2], [y + h / 2]], np.float32))

        prediction = kalman.predict()
        cv2.rectangle(frame, (int(prediction[0] - bbox[2] / 2), int(prediction[1] - bbox[3] / 2),
                              bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_bounding_box_with_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    bbox = np.array([bbox])
    bbox = cv2.BoundingRect(bbox)
    bbox = tuple(bbox)
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Object Tracking with Optical Flow', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object_with_background_subtractor(video_path)
    track_bounding_box_with_optical_flow(video_path)

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# track_obj_detect.py
### Date: 20.03.2024-18:50:32
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2

def detect_object_with_background_subtractor(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_bounding_box_with_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, bbox)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking with Optical Flow', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object_with_background_subtractor(video_path)
    track_bounding_box_with_optical_flow(video_path)

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# track_obj_track.py
### Date: 20.03.2024-18:51:34
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2

def detect_object_with_background_subtractor(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_bounding_box_with_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    tracker = cv2.TrackerMOSSE_create()
    tracker.init(frame, bbox)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Object Tracking with Optical Flow', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object_with_background_subtractor(video_path)
    track_bounding_box_with_optical_flow(video_path)

if __name__ == "__main__":
    import os
    from typing import List
    import traceback

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
# detect_track_obj.py
### Date: 20.03.2024-18:52:16
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2

def detect_object_with_background_subtractor(cap, bg_subtractor):
    ret, frame = cap.read()
    if not ret:
        return None

    fg_mask = bg_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

def track_bounding_box_with_optical_flow(frame, tracker):
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Object Tracking with Optical Flow', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
tracker = cv2.TrackerMOSSE_create()

while True:
    frame = detect_object_with_background_subtractor(cap, bg_subtractor)
    if frame is None:
        break

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```
# detect_track_obj.py
### Date: 20.03.2024-18:52:44
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2

def detect_object_with_background_subtractor(cap, bg_subtractor):
    ret, frame = cap.read()
    if not ret:
        return None

    fg_mask = bg_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

def track_bounding_box_with_optical_flow(frame, tracker):
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Object Tracking with Optical Flow', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
tracker = cv2.TrackerKCF_create()

while True:
    frame = detect_object_with_background_subtractor(cap, bg_subtractor)
    if frame is None:
        break

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```
# detect_track_obj.py
### Date: 20.03.2024-18:53:55
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2

def detect_object_with_background_subtractor(cap, bg_subtractor):
    ret, frame = cap.read()
    if not ret:
        return None

    fg_mask = bg_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

def track_bounding_box_with_optical_flow(frame, tracker):
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Object Tracking with Optical Flow', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
tracker = cv2.TrackerMOSSE_create()

while True:
    frame = detect_object_with_background_subtractor(cap, bg_subtractor)
    if frame is None:
        break

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```
# detect_track_obj.py
### Date: 20.03.2024-18:57:13
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object(video_path):
    cap = cv2.VideoCapture(video_path)
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2, 0)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    measurement = np.zeros((2, 1), np.float32)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = background_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            measurement[0, 0] = x + 0.5 * w
            measurement[1, 0] = y + 0.5 * h
            kalman_filter.correct(measurement)
            prediction = kalman_filter.predict()
            bbox = (int(prediction[0, 0] - 0.5 * w), int(prediction[1, 0] - 0.5 * h), w, h)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.imshow('Object Detection', frame)
        else:
            kalman_filter.statePost = kalman_filter.statePre
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def track_bbox(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ret, frame = cap.read()
    bbox = cv2.selectROI('Select Object to Track', frame, fromCenter=False)
    bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    roi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, _, _ = cv2.calcOpticalFlowPyrLK(roi_gray, frame_gray, None, None, **lk_params)
        p1 = p1.reshape(-1, 2)
        p0 = np.array([[bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]]], np.float32)
        p1 = np.mean(p1, axis=0, dtype=np.float32).reshape(1, 2)
        bbox = (int(p1[0, 0] - 0.5 * bbox[2]), int(p1[0, 1] - 0.5 * bbox[3]), bbox[2], bbox[3])
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object(video_path)
    track_bbox(video_path)

import os
import traceback

if __name__ == "__main__":
    videos = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            videos.append(os.path.join("./assets", path))
    for video in sorted(videos):
        try:
            track(video)
        except:
            traceback.print_exc()

```
# detect_track_obj.py
### Date: 20.03.2024-18:58:31
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object(video_path):
    cap = cv2.VideoCapture(video_path)
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2, 0)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    measurement = np.zeros((2, 1), np.float32)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = background_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            measurement[0, 0] = x + 0.5 * w
            measurement[1, 0] = y + 0.5 * h
            kalman_filter.correct(measurement)
            prediction = kalman_filter.predict()
            bbox = (int(prediction[0, 0] - 0.5 * w), int(prediction[1, 0] - 0.5 * h), w, h)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.imshow('Object Detection', frame)
        else:
            kalman_filter.statePost = kalman_filter.statePre
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def track_bbox(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ret, frame = cap.read()
    bbox = cv2.selectROI('Select Object to Track', frame, fromCenter=False)
    bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = frame_gray[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        p1, _, _ = cv2.calcOpticalFlowPyrLK(roi_gray, frame_gray, None, None, **lk_params)
        p1 = p1.reshape(-1, 2)
        p0 = np.array([[bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]]], np.float32)
        p1 = np.mean(p1, axis=0, dtype=np.float32).reshape(1, 2)
        bbox = (int(p1[0, 0] - 0.5 * bbox[2]), int(p1[0, 1] - 0.5 * bbox[3]), bbox[2], bbox[3])
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object(video_path)
    track_bbox(video_path)

import os
import traceback

if __name__ == "__main__":
    videos = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            videos.append(os.path.join("./assets", path))
    for video in sorted(videos):
        try:
            track(video)
        except:
            traceback.print_exc()

```
# detect_track_obj.py
### Date: 20.03.2024-18:59:31
### Meta: opencv-gen-test
### Purpose: coder
```py
import cv2
import numpy as np

def detect_object(video_path):
    cap = cv2.VideoCapture(video_path)
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    kalman_filter = cv2.KalmanFilter(4, 2, 0)
    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
    measurement = np.zeros((2, 1), np.float32)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = background_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            measurement[0, 0] = x + 0.5 * w
            measurement[1, 0] = y + 0.5 * h
            kalman_filter.correct(measurement)
            prediction = kalman_filter.predict()
            bbox = (int(prediction[0, 0] - 0.5 * w), int(prediction[1, 0] - 0.5 * h), w, h)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.imshow('Object Detection', frame)
        else:
            kalman_filter.statePost = kalman_filter.statePre
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def track_object(video_path):
    cap = cv2.VideoCapture(video_path)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ret, frame = cap.read()
    fg_mask = cv2.createBackgroundSubtractorMOG2()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = fg_mask.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            bbox = (x, y, w, h)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = frame_gray[y:y+h, x:x+w]
            p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            p0 = np.float32(p0).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(roi_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
                frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
            cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def track(video_path):
    detect_object(video_path)
    track_object(video_path)

import os
import traceback

if __name__ == "__main__":
    videos = []
    for path in os.listdir("./assets"):
        if path.endswith(".mp4"):
            videos.append(os.path.join("./assets", path))
    for video in sorted(videos):
        try:
            track(video)
        except:
            traceback.print_exc()

```
