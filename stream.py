"""Minimal Flask MJPEG stream: view live frames in a browser at http://<host>:<port>/."""
import time

import cv2
from flask import Flask, Response


def serve(read, port=8080, quality=70):
    """`read()` must return (frame_id, frame) like Camera.read(). Blocks until Ctrl+C."""
    app = Flask(__name__)

    def frames():
        last_id, last_t, fps = 0, time.time(), 0.0
        while True:
            fid, frame = read()
            if frame is None or fid == last_id:
                time.sleep(0.001)
                continue
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
            last_id, last_t = fid, now
            img = frame.copy()
            cv2.putText(img, f"{img.shape[1]}x{img.shape[0]} {fps:.0f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"

    @app.route("/")
    def index():
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"Livebild: http://localhost:{port}/  (Ctrl+C avslutar)", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True)
