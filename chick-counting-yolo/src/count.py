import argparse
import cv2
from ultralytics import YOLO
from utils import draw_count

def main():
    parser = argparse.ArgumentParser(description="Chick counting with YOLOv8")
    parser.add_argument("--source", type=str, default="data/videos/sample.mp4",
                        help="Path to video file or webcam index (e.g., 0)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    
    model = YOLO("yolov8n.pt")

  
    source = args.source
    cap = None
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}. If using a file, check the path. If using webcam, try --source 0")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, conf=args.conf, verbose=False)
        boxes = results[0].boxes if results and len(results) > 0 else []
        count = len(boxes) if boxes is not None else 0

        frame = draw_count(frame, count)
        cv2.imshow("Chick Counting (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
