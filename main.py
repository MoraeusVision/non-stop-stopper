"""Non-stop sorter main loop.

Threads: camera (capture) -> inference/sorting (this module's worker) -> display (main thread).
"""
import argparse
import threading
import time

import cv2
import numpy as np

import config
from camera import Camera
from geometry import Geometry
from visualizer import draw


class Pipeline:
    def __init__(self, detector, anomaly, geometry, valves):
        self.detector = detector
        self.anomaly = anomaly
        self.geometry = geometry
        self.valves = valves
        self.lock = threading.Lock()
        self.result = None  # (frame, detections, fired, fps)
        self.running = False

    def process(self, frame):
        dets = self.detector.detect(frame)
        if len(dets):
            in_zone = self.geometry.in_inspection_zone(dets)
            for i in np.flatnonzero(in_zone):
                x1, y1, x2, y2 = dets.xyxy[i].astype(int)
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                if self.anomaly.is_anomalous(crop):
                    dets.class_id[i] = config.REJECT_CLASS_ID
        fired = self.geometry.crossings(dets, config.CLASS_NAMES) if len(dets) else []
        for name in fired:
            self.valves.fire(name)
        return dets, fired

    def run(self, camera):
        last_id, last_t = 0, time.time()
        fps = 0.0
        while self.running:
            fid, frame = camera.read()
            if frame is None or fid == last_id:
                time.sleep(0.001)
                continue
            last_id = fid
            dets, fired = self.process(frame)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
            last_t = now
            with self.lock:
                self.result = (frame, dets, fired, fps)

    def latest(self):
        with self.lock:
            return self.result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpio", action="store_true", help="print valve events instead of driving GPIO")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    from patchcore_detector import PatchCoreDetector
    from rfdetr_detector import RFDETRDetector
    from valves import DummyValveController, ValveController

    detector = RFDETRDetector(
        config.RFDETR_WEIGHTS_PATH, config.RFDETR_NUM_CLASSES, config.RFDETR_INPUT_SIZE,
        config.RFDETR_CONF_THRESHOLD, config.RFDETR_CLASS_OFFSET,
    )
    anomaly = PatchCoreDetector(
        config.PATCHCORE_ENGINE_PATH, config.PATCHCORE_INPUT_SIZE, config.PATCHCORE_ANOMALY_THRESHOLD
    )
    geometry = Geometry(config.INSPECTION_ZONE, config.TRIGGER_LINES)
    valves = (DummyValveController() if args.no_gpio else
              ValveController(config.GPIO_PINS, config.VALVE_PULSE_SECONDS, config.GPIO_ACTIVE_HIGH))
    camera = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, config.CAMERA_FPS).start()

    pipe = Pipeline(detector, anomaly, geometry, valves)
    pipe.running = True
    worker = threading.Thread(target=pipe.run, args=(camera,), daemon=True)
    worker.start()

    show = config.SHOW_WINDOW and not args.no_display
    try:
        while True:
            res = pipe.latest()
            if show and res is not None:
                frame, dets, fired, fps = res
                img = draw(frame, dets, config.CLASS_NAMES, config.INSPECTION_ZONE,
                           config.TRIGGER_LINES, fired, fps)
                cv2.imshow(config.WINDOW_NAME, img)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.running = False
        worker.join(timeout=2)
        camera.stop()
        valves.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
