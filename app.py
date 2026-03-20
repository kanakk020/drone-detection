import math
import time
import threading
import os
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, send_file
from ultralytics import YOLO
import io

app = Flask(__name__)

# Basic configuration for video + detection
CONFIG = {
    "video_source": 0,              # default webcam
    "confidence_thresh": 0.55,     # minimum confidence to show detection
    "frame_size": (640, 480),      # resize frames for performance
    "output_dir": "output",
    "jpeg_quality": 60,            # compression for streaming
    "skip_frames": 3,              # skip frames to reduce load
}

RUN_DETECTION = True
CAMERA_ON = True

# Shared state between threads
state = {
    "frame": None,
    "detections": [],
    "fps": 0,
    "total_dets": 0,
    "telemetry": {},
    "lock": threading.Lock(),
}

# Fake telemetry to simulate drone data
def get_simulated_telemetry():
    t = time.time()
    return {
        "lat": round(28.6139 + math.sin(t * 0.03) * 0.0005, 6),
        "lon": round(77.2090 + math.cos(t * 0.03) * 0.0005, 6),
        "altitude": round(50 + math.sin(t * 0.1) * 5, 1),
        "speed": round(abs(math.sin(t * 0.07)) * 12, 1),
        "heading": int((t * 5) % 360),
        "battery": max(20, 100 - int((t % 120) / 1.2)),
    }

# Runs continuously in background to capture frames + detect objects
def detection_loop():
    global CAMERA_ON

    model = YOLO("yolov8n.pt")

    # Using DirectShow backend to avoid camera issues on Windows
    cap = cv2.VideoCapture(CONFIG["video_source"], cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] Camera not found")
        return

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    frame_count = 0
    prev_time = 0

    while True:

        # If camera is turned off, just reset values and wait
        if not CAMERA_ON:
            with state["lock"]:
                state["detections"] = []
                state["fps"] = 0

            time.sleep(0.2)
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, CONFIG["frame_size"])
        frame_count += 1

        # Skip some frames to keep performance smooth
        if frame_count % CONFIG["skip_frames"] != 0:
            continue

        detections = []

        # Run YOLO detection if enabled
        if RUN_DETECTION:
            results = model(frame, imgsz=320, verbose=False)

            for box in results[0].boxes:
                conf = float(box.conf[0])

                if conf < CONFIG["confidence_thresh"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]

                detections.append({
                    "label": label,
                    "conf": conf,
                    "priority": "low",
                    "distance": None,
                    "in_center": False
                })

                # Draw bounding box + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # Encode frame for streaming
        _, jpeg = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, CONFIG["jpeg_quality"]])

        # Update shared data safely
        with state["lock"]:
            state["frame"] = jpeg.tobytes()
            state["detections"] = detections
            state["fps"] = int(fps)
            state["telemetry"] = get_simulated_telemetry()

# Generator that streams frames to browser
def generate_frames():
    while True:
        with state["lock"]:
            frame = state["frame"]

        # Show blank screen if camera is off
        if not CAMERA_ON:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "CAMERA OFF", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, jpeg = cv2.imencode(".jpg", blank)
            frame = jpeg.tobytes()

        if frame is None:
            time.sleep(0.05)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        time.sleep(0.06)

# ---------------- Routes ----------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# Toggle camera ON/OFF from frontend
@app.route("/toggle_camera", methods=["POST"])
def toggle_camera():
    global CAMERA_ON
    CAMERA_ON = not CAMERA_ON
    return jsonify({"camera_on": CAMERA_ON})

# Returns current detections + FPS
@app.route("/api/detections")
def api_detections():
    with state["lock"]:
        return jsonify({
            "detections": state["detections"],
            "fps": state["fps"],
            "total_dets": len(state["detections"]),
        })

# Returns simulated telemetry data
@app.route("/api/telemetry")
def api_telemetry():
    with state["lock"]:
        return jsonify(state["telemetry"])

# Capture current frame as image
@app.route("/snapshot")
def snapshot():
    with state["lock"]:
        frame = state["frame"]

    if frame is None:
        return "No frame", 503

    return send_file(io.BytesIO(frame),
                     mimetype="image/jpeg",
                     as_attachment=True,
                     download_name="snapshot.jpg")

# Start detection thread + Flask server
if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    print("🔥 Running → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)