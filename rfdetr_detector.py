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
        class_ids = np.asarray(dets.class_id, dtype=int) - self.class_offset
        return dets[class_ids >= 0] if (class_ids < 0).any() else _with_ids(dets, class_ids)


def _with_ids(dets, class_ids):
    dets.class_id = class_ids
    return dets
