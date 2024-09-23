import os
import numpy as np
import cv2


def load_video(path, max_frames=0):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            cv2.imwrite(os.path.dirname(path) + "/" + str(len(frames)) + ".jpg", frame)
            if len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)
size = int(input("Insert the number of frames "))
symbol = input("Insert the symbol ")
filename = input("Insert the filename (without its extension) ")
load_video("data/"+ symbol + "/"  + filename + ".avi", size)