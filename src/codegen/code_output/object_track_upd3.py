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
