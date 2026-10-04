"""Central configuration. Calibrate zones/lines with the visualization window."""

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 120

# --- Classes ---
# Index = class id output by RF-DETR. Reject is assigned by PatchCore.
COLOR_CLASSES = ["red", "orange", "yellow", "green", "blue", "brown"]
REJECT_CLASS_NAME = "reject"
CLASS_NAMES = COLOR_CLASSES + [REJECT_CLASS_NAME]
REJECT_CLASS_ID = len(COLOR_CLASSES)

# --- Models ---
# RF-DETR: fine-tuned RFDETRBase checkpoint (.pth) loaded directly by the `rfdetr` package (no PyCUDA).
# (rfdetr loads .pth checkpoints only; export_rfdetr_trt.py can build a .trt engine separately.)
RFDETR_WEIGHTS_PATH = "models/rfdetr_base_checkpoint.pth"
RFDETR_NUM_CLASSES = len(COLOR_CLASSES)
# PatchCore: TensorRT engine
PATCHCORE_ENGINE_PATH = "models/patchcore.engine"
RFDETR_INPUT_SIZE = 560           # square input, pixels
RFDETR_CONF_THRESHOLD = 0.5
# Offset subtracted from predicted class id (1 if dataset class ids start at 1)
RFDETR_CLASS_OFFSET = 0
PATCHCORE_INPUT_SIZE = 224
PATCHCORE_ANOMALY_THRESHOLD = 0.5  # score above this => defective

# --- Geometry (pixel coordinates in camera frame) ---
INSPECTION_ZONE = [(0, 200), (200, 200), (200, 520), (0, 520)]

# Trigger lines: name -> ((x1, y1), (x2, y2)); name is a class name.
# Belt assumed to travel left -> right, lines are vertical.
TRIGGER_LINES = {
    "reject": ((400, 150), (400, 570)),
    "red": ((500, 150), (500, 570)),
    "orange": ((600, 150), (600, 570)),
    "yellow": ((700, 150), (700, 570)),
    "green": ((800, 150), (800, 570)),
    "blue": ((900, 150), (900, 570)),
    "brown": ((1000, 150), (1000, 570)),
}

# --- GPIO (BOARD pin numbering) ---
GPIO_PINS = {
    "reject": 7,
    "red": 11,
    "orange": 12,
    "yellow": 13,
    "green": 15,
    "blue": 16,
    "brown": 18,
}
VALVE_PULSE_SECONDS = 0.05
GPIO_ACTIVE_HIGH = True

# --- Display ---
SHOW_WINDOW = True
WINDOW_NAME = "non-stop-stopper"
