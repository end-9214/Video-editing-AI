import os
import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import shutil

model = YOLO("yolov8n.pt")

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light"
}

object_to_crop = input("Enter the object you want to track (e.g., 'person', 'car'): ").strip().lower()

object_class_id = None
for class_id, class_name in COCO_CLASSES.items():
    if class_name == object_to_crop:
        object_class_id = class_id
        break

if object_class_id is None:
    print("❌ Invalid object! Exiting...")
    exit()

video_path = './Videos/13035233_2160_3840_30fps.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_folder = 'cropped_frames'
os.makedirs(output_folder, exist_ok=True)

frame_skip = 4
frame_count = 0
saved_frames = 0

history_length = 13
bbox_history = []
fixed_width, fixed_height = None, None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    if frame_count % frame_skip == 0:
        results = model(frame)
        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if cls == object_class_id:
                    detections.append(box)

        if detections:
            detections = np.array(detections)
            x1 = int(np.min(detections[:, 0]))
            y1 = int(np.min(detections[:, 1]))
            x2 = int(np.max(detections[:, 2]))
            y2 = int(np.max(detections[:, 3]))
            bbox_history.append((x1, y1, x2, y2))

        if len(bbox_history) > history_length:
            bbox_history.pop(0)

    if bbox_history:
        avg_x1 = int(np.mean([bbox[0] for bbox in bbox_history]))
        avg_y1 = int(np.mean([bbox[1] for bbox in bbox_history]))
        avg_x2 = int(np.mean([bbox[2] for bbox in bbox_history]))
        avg_y2 = int(np.mean([bbox[3] for bbox in bbox_history]))

        bbox_width = avg_x2 - avg_x1
        bbox_height = avg_y2 - avg_y1

        target_aspect_ratio = 9 / 16
        if (bbox_width / bbox_height) > target_aspect_ratio:
            new_height = int(bbox_width / target_aspect_ratio)
            y_center = (avg_y1 + avg_y2) // 2
            avg_y1 = max(0, y_center - new_height // 2)
            avg_y2 = min(original_height, y_center + new_height // 2)
        else:
            new_width = int(bbox_height * target_aspect_ratio)
            x_center = (avg_x1 + avg_x2) // 2
            avg_x1 = max(0, x_center - new_width // 2)
            avg_x2 = min(original_width, x_center + new_width // 2)

        cropped_frame = frame[avg_y1:avg_y2, avg_x1:avg_x2]

        if fixed_width is None or fixed_height is None:
            fixed_height, fixed_width = cropped_frame.shape[:2]

        cropped_frame_resized = cv2.resize(cropped_frame, (fixed_width, fixed_height))

        frame_filename = os.path.join(output_folder, f"frame_{saved_frames:06d}.jpg")
        cv2.imwrite(frame_filename, cropped_frame_resized)
        saved_frames += 1

cap.release()
cv2.destroyAllWindows()
print(f"✅ Saved {saved_frames} smoothed cropped frames in '{output_folder}'")

output_video = "cropped_video.mp4"
frame_files = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".jpg")])

if frame_files:
    clip = ImageSequenceClip(frame_files, fps=fps)
    clip.write_videofile(output_video, codec="libx264", fps=fps)
    print(f"🎥 Cropped video saved as '{output_video}'")

    try:
        shutil.rmtree(output_folder)  
        print(f"🗑️ Deleted cropped frames folder: '{output_folder}'")
    except Exception as e:
        print(f"⚠️ Error deleting folder: {e}")

else:
    print("❌ No frames found to create a video!")