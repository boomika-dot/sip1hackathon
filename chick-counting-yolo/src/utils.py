import os
import cv2

def ensure_path(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def draw_count(frame, count: int):
    cv2.putText(frame, f"Chick Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame
