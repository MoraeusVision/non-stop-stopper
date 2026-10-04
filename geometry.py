"""Zone and line-crossing logic built on supervision."""
import numpy as np
import supervision as sv


class Geometry:
    def __init__(self, inspection_zone, trigger_lines):
        self.zone = sv.PolygonZone(
            polygon=np.array(inspection_zone, dtype=np.int32),
            triggering_anchors=(sv.Position.CENTER,),
        )
        self.line_specs = trigger_lines
        self.lines = {
            name: sv.LineZone(
                start=sv.Point(*a), end=sv.Point(*b), triggering_anchors=(sv.Position.CENTER,)
            )
            for name, (a, b) in trigger_lines.items()
        }

    def in_inspection_zone(self, detections):
        """Boolean mask of detections whose centre is inside the inspection zone."""
        if len(detections) == 0:
            return np.zeros(0, dtype=bool)
        return self.zone.trigger(detections)

    def crossings(self, detections, class_names):
        """Return names of lines crossed this frame by a detection of matching class.

        Without tracking, each detection gets a per-frame pseudo tracker id by
        sorted position along the belt (candy are in a single row), which is
        sufficient for LineZone to see a centre move across a line.
        """
        fired = []
        for name, line in self.lines.items():
            sel = np.array([class_names[c] == name for c in detections.class_id], dtype=bool)
            if not sel.any():
                continue
            sub = detections[sel]
            sub.tracker_id = self._pseudo_ids(sub)
            crossed_in, crossed_out = line.trigger(sub)
            if crossed_in.any() or crossed_out.any():
                fired.append(name)
        return fired

    @staticmethod
    def _pseudo_ids(dets):
        # Stable-ish ids: quantised centre along the belt's dominant axis.
        # LineZone needs consistent ids between frames; use rank by position.
        order = np.argsort(np.argsort(dets.xyxy[:, 0]))
        return order.astype(int)
