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
