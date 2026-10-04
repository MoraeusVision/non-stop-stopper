"""PatchCore TensorRT inference: anomaly (defect) detection on candy crops.

The raw TensorRT/PyCUDA wrapper lives here only; RF-DETR uses the `rfdetr` package instead.
"""
import cv2
import numpy as np


# --- TensorRT 10 tensor API + pycuda, static shapes only ---

class TRTEngine:
    def __init__(self, engine_path):
        import pycuda.autoinit  # noqa: F401  (creates CUDA context)
        import pycuda.driver as cuda
        import tensorrt as trt

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            if -1 in shape:
                raise RuntimeError(f"Dynamic shape not supported for {name}: {shape}")
            host = cuda.pagelocked_empty(shape, dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            entry = {"name": name, "host": host, "dev": dev}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

    @property
    def input_shape(self):
        return self.inputs[0]["host"].shape

    def infer(self, array):
        """Run inference on a single input; returns list of output arrays."""
        cuda = self._cuda
        inp = self.inputs[0]
        np.copyto(inp["host"], np.asarray(array, dtype=inp["host"].dtype).reshape(inp["host"].shape))
        cuda.memcpy_htod_async(inp["dev"], inp["host"], self.stream)
        self.context.execute_async_v3(self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["dev"], self.stream)
        self.stream.synchronize()
        return [o["host"].copy() for o in self.outputs]


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
