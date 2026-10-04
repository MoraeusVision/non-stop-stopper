"""Export a fine-tuned RFDETRBase checkpoint to a TensorRT engine (.trt).

rfdetr_detector.py calls ensure_engine() automatically when the engine is missing; this script
lets you run the same export manually. Run on the Jetson itself (engines are tied to the GPU and
TensorRT version):

    pip install "rfdetr[tensorrt]"
    python export_rfdetr_trt.py [--weights models/rfdetr_base_checkpoint.pth] [--engine models/rfdetr.trt]
"""
import argparse
import os
import shutil

import config


def ensure_engine(engine_path, weights_path, num_classes, resolution, fp16=True, force=False):
    """Return engine_path, building it from the RFDETRBase checkpoint if it doesn't exist."""
    if os.path.isfile(engine_path) and not force:
        return engine_path
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"TensorRT engine '{engine_path}' not found and no checkpoint at '{weights_path}' to build it from"
        )
    from rfdetr import RFDETRBase

    print(f"[rfdetr] Exporting TensorRT engine from {weights_path} ...")
    out_dir = os.path.dirname(engine_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    model = RFDETRBase(pretrain_weights=weights_path, num_classes=num_classes, resolution=resolution)
    exported = str(
        model.export(
            output_dir=out_dir,
            format="tensorrt",
            fp16=fp16,
            output_name=os.path.splitext(os.path.basename(engine_path))[0],
        )
    )
    if os.path.abspath(exported) != os.path.abspath(engine_path):
        shutil.move(exported, engine_path)
    print(f"[rfdetr] TensorRT engine written to {engine_path}")
    return engine_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=config.RFDETR_WEIGHTS_PATH)
    ap.add_argument("--engine", default=config.RFDETR_ENGINE_PATH)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild even if the engine exists")
    args = ap.parse_args()
    ensure_engine(
        args.engine, args.weights, config.RFDETR_NUM_CLASSES, config.RFDETR_INPUT_SIZE,
        fp16=not args.no_fp16, force=args.force,
    )


if __name__ == "__main__":
    main()
