from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = 'C:/Users/DELL-7373/Desktop/Projects/Video-editing-AI/Videos/13035233_2160_3840_30fps.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Get video properties (original width, height, FPS)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video path
output_path = 'C:/Users/DELL-7373/Desktop/Projects/Video-editing-AI/Videos/cropped_video.mp4'  # Choose your output path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 (adjust if needed)
out = None  # Initialize video writer *outside* the loop
frame_width = None
frame_height = None


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)

    all_person_boxes = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if cls == 0:  # Person class
                all_person_boxes.append(box)

    if all_person_boxes:
        all_person_boxes = np.array(all_person_boxes)
        min_x = int(np.min(all_person_boxes[:, 0]))
        min_y = int(np.min(all_person_boxes[:, 1]))
        max_x = int(np.max(all_person_boxes[:, 2]))
        max_y = int(np.max(all_person_boxes[:, 3]))

        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        bbox_aspect_ratio = bbox_width / bbox_height

        target_aspect_ratio = 9/16
        if bbox_aspect_ratio > target_aspect_ratio:
            new_height = int(bbox_width / target_aspect_ratio)
            y_center = (min_y + max_y) // 2
            min_y = max(0, y_center - new_height // 2)
            max_y = min(original_height, y_center + new_height // 2)
        else:
            new_width = int(bbox_height * target_aspect_ratio)
            x_center = (min_x + max_x) // 2
            min_x = max(0, x_center - new_width // 2)
            max_x = min(original_width, x_center + new_width // 2)

        cropped_frame = frame[min_y:max_y, min_x:max_x]

        if out is None:  # Initialize VideoWriter only once
            frame_height, frame_width, _ = cropped_frame.shape  # Get cropped frame dimensions
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        out.write(cropped_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if out is not None:
    out.release()  # Release the VideoWriter
cv2.destroyAllWindows()