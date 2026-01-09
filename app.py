import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import os

# -----------------------------
# Load ONNX model
# -----------------------------
model = YOLO("best.onnx")

IMG_SIZE = 736

# -----------------------------
# Per-class confidence thresholds
# -----------------------------
CLASS_THRESHOLDS = {
    "Fire": 0.30,
    "gun": 0.70,
    "knife": 0.28,
    "person": 0.22
}

# -----------------------------
# Per-class colors (BGR)
# -----------------------------
CLASS_COLORS = {
    "Fire": (0, 0, 255),
    "gun": (255, 0, 0),
    "knife": (0, 165, 255),
    "person": (0, 255, 0)
}

# -----------------------------
# Video detection function
# -----------------------------
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=0.20,   # low global confidence
            save=False,
            verbose=False
        )

        result = results[0]
        annotated = frame.copy()

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])

                # Per-class confidence filtering
                required_conf = CLASS_THRESHOLDS.get(cls_name, 0.3)
                if conf < required_conf:
                    continue

                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Color per class
                color = CLASS_COLORS.get(cls_name, (255, 255, 255))

                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{cls_name} {int(conf * 100)}%"
                (w, h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                cv2.rectangle(
                    annotated,
                    (x1, y1 - h - 10),
                    (x1 + w + 4, y1),
                    color,
                    -1
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        out.write(annotated)

    cap.release()
    out.release()

    return output_path

# -----------------------------
# Gradio UI (Video)
# -----------------------------
demo = gr.Interface(
    fn=detect_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Video(label="Detection Output"),
    title="Weapon Detection – YOLO ONNX (736)",
    description=(
        "Per-class confidence thresholds:\n"
        "Fire = 0.30 | Gun = 0.70 | Knife = 0.28 | Person = 0.22\n\n"
        "Each class is displayed with a unique color."
    )
)

demo.launch()