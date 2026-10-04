"""RF-DETR inference via the `rfdetr` package (RFDETRBase): candy detection + colour classification."""
import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRBase


class RFDETRDetector:
    def __init__(self, weights_path, num_classes, resolution=560, conf_threshold=0.5, class_offset=0):
        self.model = RFDETRBase(
            pretrain_weights=weights_path, num_classes=num_classes, resolution=resolution
        )
        try:
            self.model.optimize_for_inference()
        except Exception as exc:  # optimisation is optional
            print(f"[rfdetr] optimize_for_inference skipped: {exc}")
        self.conf = conf_threshold
        self.class_offset = class_offset

    def detect(self, frame):
        """Return sv.Detections in frame pixel coords with class_id = colour id."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = self.model.predict(rgb, threshold=self.conf)
        if len(dets) == 0:
            return sv.Detections.empty()
        dets.class_id = np.asarray(dets.class_id, dtype=int) - self.class_offset
        return dets[dets.class_id >= 0]
