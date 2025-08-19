import sys
from ultralytics import YOLO

def run_detection(source):
    
    model = YOLO("yolov8n.pt")
   
    results = model.predict(source=source, show=True, save=True)
    return results

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "data/images/sample.jpg"
    print(f"[INFO] Running detection on: {source}")
    run_detection(source)
