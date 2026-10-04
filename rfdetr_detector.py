"""RF-DETR TensorRT inference: candy detection + colour classification."""
import cv2
import numpy as np
import supervision as sv

from trt_engine import TRTEngine

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RFDETRDetector:
    def __init__(self, engine_path, input_size=560, conf_threshold=0.5, class_offset=0):
        self.engine = TRTEngine(engine_path)
        self.size = input_size
        self.conf = conf_threshold
        self.class_offset = class_offset

    def _preprocess(self, frame):
        img = cv2.resize(frame, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return img.transpose(2, 0, 1)[None]

    def detect(self, frame):
        """Return sv.Detections in frame pixel coords with class_id = colour id."""
        h, w = frame.shape[:2]
        outputs = self.engine.infer(self._preprocess(frame))
        # Identify outputs by last dim: boxes (…,4) vs logits (…,C)
        boxes = next(o for o in outputs if o.shape[-1] == 4)[0]
        logits = next(o for o in outputs if o.shape[-1] != 4)[0]
        probs = _sigmoid(logits)
        class_ids = probs.argmax(axis=1)
        scores = probs.max(axis=1)
        keep = scores >= self.conf
        boxes, class_ids, scores = boxes[keep], class_ids[keep], scores[keep]
        if len(boxes) == 0:
            return sv.Detections.empty()
        cx, cy, bw, bh = boxes.T  # normalized cxcywh
        xyxy = np.stack(
            [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1
        ).astype(np.float32)
        class_ids = class_ids - self.class_offset
        valid = class_ids >= 0
        return sv.Detections(
            xyxy=xyxy[valid], confidence=scores[valid].astype(np.float32), class_id=class_ids[valid].astype(int)
        )
