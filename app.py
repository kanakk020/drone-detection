# ==========================================================
# Project Name  : AI Drone Vision — Flask Web Dashboard
# Author        : Kanak Saini
# Description   : Flask server that streams YOLO-processed
#                 video feed and exposes detection data via
#                 a JSON API for the live web dashboard.
# ==========================================================

import cv2
import math
import time
import threading
import datetime
import os
import json
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__)

# -------------------- CONFIGURATION --------------------
CONFIG = {
    "video_source":      0,
    "confidence_thresh": 0.55,
    "target_zone_pct":   (0.35, 0.65),
    "frame_size":        (640, 480),
    "alert_log":         True,
    "output_dir":        "output",
    "alert_cooldown_s":  2.0,
    "font":              cv2.FONT_HERSHEY_SIMPLEX,
}

CLASS_PRIORITY = {
    "person": 10, "car": 7, "truck": 6,
    "motorcycle": 6, "bicycle": 5, "bus": 5,
    "boat": 4, "dog": 3, "cat": 3,
}

TIER_COLORS = {
    "critical": (0, 0, 255),
    "high":     (0, 165, 255),
    "medium":   (0, 255, 255),
    "low":      (0, 255, 0),
}

# -------------------- SHARED STATE --------------------
# Accessed by both the detection thread and Flask routes
state = {
    "frame":        None,
    "detections":   [],
    "fps":          0,
    "alert_count":  0,
    "total_dets":   0,
    "telemetry":    {},
    "alert_events": [],
    "lock":         threading.Lock(),
}

# -------------------- HELPERS --------------------
def get_simulated_telemetry():
    t = time.time()
    return {
        "lat":      round(28.6139 + math.sin(t * 0.03) * 0.0005, 6),
        "lon":      round(77.2090 + math.cos(t * 0.03) * 0.0005, 6),
        "altitude": round(50 + math.sin(t * 0.1) * 5, 1),
        "speed":    round(abs(math.sin(t * 0.07)) * 12, 1),
        "heading":  int((t * 5) % 360),
        "battery":  max(20, 100 - int((t % 120) / 1.2)),
    }

def estimate_distance(box_h, frame_h, known_h=1.75):
    if box_h <= 0:
        return None
    focal = frame_h / (2 * math.tan(math.radians(30)))
    return round((known_h * focal) / box_h, 1)

def classify_priority(class_name, in_center):
    score = CLASS_PRIORITY.get(class_name.lower(), 1)
    if in_center:
        score += 5
    if score >= 14: return "critical"
    if score >= 9:  return "high"
    if score >= 5:  return "medium"
    return "low"

def draw_detection(frame, det):
    x1, y1, x2, y2 = det["coords"]
    color = TIER_COLORS[det["priority"]]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner accents
    cs = 12
    for px, py, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x2,y2,-1,-1),(x1,y2,1,-1)]:
        cv2.line(frame, (px, py), (px + sx*cs, py), color, 2)
        cv2.line(frame, (px, py), (px, py + sy*cs), color, 2)

    # Label
    dist_str = f"  {det['distance']}m" if det["distance"] else ""
    text = f"{det['label'].upper()}  {det['conf']:.0%}{dist_str}"
    (tw, th), _ = cv2.getTextSize(text, CONFIG["font"], 0.45, 1)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, ty-th-4), (x1+tw+8, ty+2), color, -1)
    cv2.putText(frame, text, (x1+4, ty-2), CONFIG["font"], 0.45, (0,0,0), 1)

    if det["priority"] in ("critical", "high"):
        badge = "TARGET LOCK" if det["priority"] == "critical" else "HIGH PRIORITY"
        cv2.putText(frame, badge, (x1, y2+16), CONFIG["font"], 0.42, color, 1)

def draw_hud(frame, fps, telem, n_dets, alerts):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,48), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "AI DRONE VISION  v2.0",
                (10, 20), CONFIG["font"], 0.5, (200,200,200), 1)
    fps_col = (0,255,120) if fps>=20 else (0,165,255) if fps>=10 else (0,0,255)
    cv2.putText(frame, f"FPS:{int(fps)}", (w-80,20), CONFIG["font"], 0.5, fps_col, 1)
    cv2.putText(frame, f"Dets:{n_dets}  Alerts:{alerts}",
                (10, 40), CONFIG["font"], 0.4, (180,180,180), 1)

    # Telemetry bottom-left
    lines = [
        f"LAT {telem['lat']:.5f}", f"LON {telem['lon']:.5f}",
        f"ALT {telem['altitude']}m", f"SPD {telem['speed']}m/s",
        f"HDG {telem['heading']:03d}°",
    ]
    batt = telem["battery"]
    bc = (0,220,80) if batt>50 else (0,165,255) if batt>25 else (0,0,255)
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (10, h-100+i*16), CONFIG["font"], 0.33, (160,200,160), 1)
    cv2.putText(frame, f"BAT {batt}%", (10, h-100+5*16), CONFIG["font"], 0.33, bc, 1)

    # Center reticle
    cx, cy = w//2, h//2
    s = 16
    for dx, dy in [(-1,-1),(1,-1),(1,1),(-1,1)]:
        cv2.line(frame,(cx+dx*s,cy+dy*s),(cx+dx*(s+12),cy+dy*s),(0,200,255),1)
        cv2.line(frame,(cx+dx*s,cy+dy*s),(cx+dx*s,cy+dy*(s+12)),(0,200,255),1)

    # Center zone dashed lines
    zl = int(w * CONFIG["target_zone_pct"][0])
    zr = int(w * CONFIG["target_zone_pct"][1])
    for y in range(0, h, 14):
        cv2.line(frame, (zl,y), (zl,min(y+7,h)), (0,180,220), 1)
        cv2.line(frame, (zr,y), (zr,min(y+7,h)), (0,180,220), 1)


# -------------------- DETECTION THREAD --------------------
def detection_loop():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(CONFIG["video_source"])
    if not cap.isOpened():
        print("[WARN] No camera found. Detection loop idle.")
        return

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    alert_log_path = os.path.join(
        CONFIG["output_dir"],
        f"alerts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    alert_events = []
    last_alert_t = 0

    prev_time = 0
    alert_count = 0
    total_dets = 0

    priority_order = {"low":0,"medium":1,"high":2,"critical":3}

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video file
            continue

        frame = cv2.resize(frame, CONFIG["frame_size"])
        h, w = frame.shape[:2]

        results = model(frame, verbose=False)
        boxes = results[0].boxes
        telem = get_simulated_telemetry()
        detections = []

        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONFIG["confidence_thresh"]:
                continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls = model.names[int(box.cls[0])]
            cx = (x1+x2)/2
            in_c = CONFIG["target_zone_pct"][0]*w < cx < CONFIG["target_zone_pct"][1]*w
            dist = estimate_distance(y2-y1, h)
            pri  = classify_priority(cls, in_c)

            detections.append({
                "coords":    (x1,y1,x2,y2),
                "conf":      conf,
                "label":     cls,
                "distance":  dist,
                "priority":  pri,
                "in_center": in_c,
            })

            # Alert logging
            if pri in ("critical","high"):
                now = time.time()
                if now - last_alert_t >= CONFIG["alert_cooldown_s"]:
                    last_alert_t = now
                    alert_count += 1
                    ev = {
                        "timestamp":  datetime.datetime.now().isoformat(),
                        "class":      cls,
                        "confidence": round(conf,3),
                        "distance_m": dist,
                        "priority":   pri,
                        "gps": {"lat": telem["lat"], "lon": telem["lon"], "alt": telem["altitude"]},
                    }
                    alert_events.append(ev)
                    with open(alert_log_path,"w") as f:
                        json.dump(alert_events, f, indent=2)

        total_dets += len(detections)
        detections.sort(key=lambda d: priority_order[d["priority"]])

        for det in detections:
            draw_detection(frame, det)

        curr_time = time.time()
        fps = 1/(curr_time-prev_time) if prev_time else 0
        prev_time = curr_time

        draw_hud(frame, fps, telem, len(detections), alert_count)

        # Encode frame to JPEG
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # Serialisable detections for JSON API
        api_dets = [
            {
                "label":    d["label"],
                "conf":     round(d["conf"], 3),
                "priority": d["priority"],
                "distance": d["distance"],
                "in_center": d["in_center"],
            }
            for d in detections
        ]

        with state["lock"]:
            state["frame"]         = jpeg.tobytes()
            state["detections"]    = api_dets
            state["fps"]           = int(fps)
            state["alert_count"]   = alert_count
            state["total_dets"]    = total_dets
            state["telemetry"]     = telem
            state["alert_events"]  = list(alert_events)


# -------------------- FLASK ROUTES --------------------
def generate_frames():
    """MJPEG stream generator."""
    while True:
        with state["lock"]:
            frame = state["frame"]
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.03)   # ~30 fps cap to browser


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/detections")
def api_detections():
    with state["lock"]:
        return jsonify({
            "detections":  state["detections"],
            "fps":         state["fps"],
            "alert_count": state["alert_count"],
            "total_dets":  state["total_dets"],
        })


@app.route("/api/alerts")
def api_alerts():
    with state["lock"]:
        return jsonify(list(reversed(state["alert_events"])))


@app.route("/snapshot")
def snapshot():
    with state["lock"]:
        frame = state["frame"]
    if frame is None:
        return "No frame available", 503
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CONFIG["output_dir"], f"snapshot_{ts}.jpg")
    with open(path, "wb") as f:
        f.write(frame)
    from flask import send_file
    import io
    return send_file(io.BytesIO(frame), mimetype="image/jpeg",
                     as_attachment=True, download_name=f"snapshot_{ts}.jpg")


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    print("[INFO] Dashboard → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)