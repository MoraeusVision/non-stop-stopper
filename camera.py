"""Threaded OpenCV camera capture that always exposes the latest frame."""
import threading

import cv2


class Camera:
    def __init__(self, index=0, width=1280, height=720, fps=120):
        self._cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
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
