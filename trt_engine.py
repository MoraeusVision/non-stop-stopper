"""Minimal TensorRT engine wrapper (TensorRT 10 tensor API + pycuda)."""
import numpy as np


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
