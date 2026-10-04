"""RF-DETR inference: candy detection + colour classification.

Uses the TensorRT engine exported by rfdetr (RFDETRBase.export) by default. If the engine is
missing it is built automatically from the RFDETRBase checkpoint (see export_rfdetr_trt.py).
"""
import cv2
import numpy as np
import supervision as sv
import torch

from export_rfdetr_trt import ensure_engine

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class RFDETRDetector:
    def __init__(self, engine_path, weights_path, num_classes, resolution=560,
                 conf_threshold=0.5, class_offset=0, max_detections=50):
        from rfdetr.export._tensorrt.inference import TRTInference

        engine_path = ensure_engine(engine_path, weights_path, num_classes, resolution)
        self.trt = TRTInference(engine_path, sync_mode=True)
        self.input_name = self.trt.input_names[0]
        self.dtype = self.trt.bindings[self.input_name].dtype
        self.device = self.trt.engine_device
        self.size = resolution
        self.conf = conf_threshold
        self.class_offset = class_offset
        self.max_detections = max_detections
        self._mean = _MEAN.to(self.device)
        self._std = _STD.to(self.device)

    def _preprocess(self, frame):
        rgb = cv2.cvtColor(cv2.resize(frame, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return ((x - self._mean) / self._std).to(self.dtype).contiguous()

    def detect(self, frame):
        """Return sv.Detections in frame pixel coords with class_id = colour id."""
        h, w = frame.shape[:2]
        out = self.trt({self.input_name: self._preprocess(frame)})
        logits, boxes = out["labels"][0].float(), out["dets"][0].float()  # (Q, C), (Q, 4) cxcywh normalised
        prob = logits.sigmoid().flatten()
        scores, idx = prob.topk(min(self.max_detections, prob.numel()))
        keep = scores >= self.conf
        scores, idx = scores[keep], idx[keep]
        if scores.numel() == 0:
            return sv.Detections.empty()
        query, cls = idx // logits.shape[1], idx % logits.shape[1]
        cx, cy, bw, bh = boxes[query].unbind(-1)
        scale = torch.tensor([w, h, w, h], device=boxes.device)
        xyxy = torch.stack(
            [cx - bw.clamp(min=0) / 2, cy - bh.clamp(min=0) / 2,
             cx + bw.clamp(min=0) / 2, cy + bh.clamp(min=0) / 2], dim=-1) * scale
        class_id = cls.cpu().numpy().astype(int) - self.class_offset
        dets = sv.Detections(
            xyxy=xyxy.cpu().numpy().astype(np.float32),
            confidence=scores.cpu().numpy().astype(np.float32),
            class_id=class_id,
        )
        return dets[class_id >= 0]
