"""PatchCore TensorRT inference: anomaly (defect) detection on candy crops."""
import cv2
import numpy as np

from trt_engine import TRTEngine

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PatchCoreDetector:
    def __init__(self, engine_path, input_size=224, threshold=0.5):
        self.engine = TRTEngine(engine_path)
        self.size = input_size
        self.threshold = threshold

    def _preprocess(self, crop):
        img = cv2.resize(crop, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((img - _MEAN) / _STD).transpose(2, 0, 1)[None]

    def score(self, crop):
        """Anomaly score = max over the model's output (score or anomaly map)."""
        outputs = self.engine.infer(self._preprocess(crop))
        return float(max(np.max(o) for o in outputs))

    def is_anomalous(self, crop):
        if crop is None or crop.size == 0:
            return False
        return self.score(crop) > self.threshold
