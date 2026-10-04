"""Threaded OpenCV camera capture that always exposes the latest frame."""
import threading

import cv2


class Camera:
    def __init__(self, index=0, width=1280, height=720, fps=120):
        self._cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        # YUYV is limited to ~10 FPS at 720p over USB; MJPG gives 30 FPS
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame = None
        self._id = 0
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                continue
            with self._lock:
                self._frame = frame
                self._id += 1

    def read(self):
        """Return (frame_id, latest frame) or (0, None)."""
        with self._lock:
            return self._id, self._frame

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        self._cap.release()


if __name__ == "__main__":
    import argparse
    import os
    import time

    import config

    ap = argparse.ArgumentParser(description="Camera test")
    ap.add_argument("--web", action="store_true", help="stream to browser (default when no display)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    web = args.web or (not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"))

    cam = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, config.CAMERA_FPS).start()
    try:
        if web:
            from stream import serve
            serve(cam.read, args.port)
        last_id, last_t, fps = 0, time.time(), 0.0
        while True:
            fid, frame = cam.read()
            if frame is None or fid == last_id:
                time.sleep(0.001)
                continue
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
            last_id, last_t = fid, now
            cv2.putText(frame, f"{frame.shape[1]}x{frame.shape[0]} {fps:.0f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("camera test", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        if not web:
            cv2.destroyAllWindows()
