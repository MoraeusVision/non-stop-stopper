# non-stop-stopper
Sorts Non-Stop candy by colour and removes defective ones (Jetson Orin Nano).

Pipeline: camera -> RF-DETR (`rfdetr` RFDETRBase) -> PatchCore (TensorRT) on candies in the
inspection zone (anomaly => `reject`) -> `supervision` line crossing -> GPIO valve pulse.

## Setup
    pip install -r requirements.txt   # plus tensorrt, pycuda, Jetson.GPIO from JetPack
Edit `config.py` (model paths, GPIO pins, inspection zone, 7 trigger lines), then:

    python main.py            # q / Esc closes the calibration window
    python main.py --no-gpio  # dry run, prints valve events

Modules: `camera.py`, `rfdetr_detector.py`, `patchcore_detector.py`, `trt_engine.py` (PatchCore),
`geometry.py`, `valves.py`, `visualizer.py`, `main.py`.
