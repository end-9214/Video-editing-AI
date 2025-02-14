from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=2)  # max_age increased to remember object longer

# Ask user for object to track
COCO_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane"}
object_to_crop = input("Enter the object you want to track (e.g., 'person', 'car'): ").strip().lower()

# Find object class ID
object_class_id = None
for class_id, class_name in COCO_CLASSES.items():
    if class_name == object_to_crop:
        object_class_id = class_id
        break

if object_class_id is None:
    print("Invalid object! Exiting...")
    exit()

# Open video
video_path = 'C:/Users/DELL-7373/Desktop/Projects/Video-editing-AI/Videos/13035233_2160_3840_30fps.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
output_path = 'cropped_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

# Store the last known bounding box
last_bbox = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO object detection
    results = model(frame)

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            if cls == object_class_id:  # If detected object is the one we want
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2, y2], conf, cls))  # Append bounding box

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Find best tracking box
    best_track = None
    for track in tracks:
        if track.is_confirmed():
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            best_track = (x1, y1, x2, y2)
            break

    # Use the last known bounding box if tracking fails
    if best_track:
        last_bbox = best_track  # Update last known bounding box
    elif last_bbox:
        x1, y1, x2, y2 = last_bbox  # Use last known bounding box

    if last_bbox:
        x1, y1, x2, y2 = last_bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_aspect_ratio = bbox_width / bbox_height

        # Maintain 9:16 aspect ratio
        target_aspect_ratio = 9 / 16
        if bbox_aspect_ratio > target_aspect_ratio:
            new_height = int(bbox_width / target_aspect_ratio)
            y_center = (y1 + y2) // 2
            y1 = max(0, y_center - new_height // 2)
            y2 = min(original_height, y_center + new_height // 2)
        else:
            new_width = int(bbox_height * target_aspect_ratio)
            x_center = (x1 + x2) // 2
            x1 = max(0, x_center - new_width // 2)
            x2 = min(original_width, x_center + new_width // 2)

        # Crop frame
        cropped_frame = frame[y1:y2, x1:x2]

        # Initialize writer if not already
        if out is None:
            frame_height, frame_width, _ = cropped_frame.shape
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Ensure valid frame before writing
        if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
            out.write(cropped_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if out:
    out.release()
    print(f"✅ Cropped video saved: {output_path}")
else:
    print("❌ No frames saved!")

cv2.destroyAllWindows()
