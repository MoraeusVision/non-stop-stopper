"""Clean OpenCV overlay for calibration."""
import cv2
import numpy as np

_COLORS = {
    "red": (0, 0, 255), "orange": (0, 140, 255), "yellow": (0, 255, 255),
    "green": (0, 200, 0), "blue": (255, 100, 0), "brown": (19, 69, 139),
    "reject": (255, 0, 255),
}


def draw(frame, detections, class_names, inspection_zone, trigger_lines, fired=(), fps=None):
    out = frame.copy()
    cv2.polylines(out, [np.array(inspection_zone, dtype=np.int32)], True, (255, 255, 0), 2)
    cv2.putText(out, "inspection", tuple(inspection_zone[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    for name, (a, b) in trigger_lines.items():
        color = _COLORS.get(name, (255, 255, 255))
        cv2.line(out, a, b, color, 4 if name in fired else 2)
        cv2.putText(out, name, (a[0] + 4, a[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for xyxy, cid in zip(detections.xyxy, detections.class_id):
        name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        color = _COLORS.get(name, (255, 255, 255))
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, name, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if fps is not None:
        cv2.putText(out, f"{fps:.0f} fps", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
