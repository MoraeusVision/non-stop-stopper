"""Optional offline step: export a fine-tuned RFDETRBase checkpoint to a TensorRT engine (.trt).

Run on the Jetson itself (engines are tied to the GPU + TensorRT version):

    pip install "rfdetr[tensorrt]"
    python export_rfdetr_trt.py [--weights models/rfdetr_base_checkpoint.pth] [--out models]

The main pipeline (main.py) does not use this; it loads the checkpoint with RFDETRBase.
"""
import argparse

from rfdetr import RFDETRBase

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=config.RFDETR_WEIGHTS_PATH)
    ap.add_argument("--out", default="models")
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()

    model = RFDETRBase(
        pretrain_weights=args.weights,
        num_classes=config.RFDETR_NUM_CLASSES,
        resolution=config.RFDETR_INPUT_SIZE,
    )
    path = model.export(output_dir=args.out, format="tensorrt", fp16=not args.no_fp16)
    print(f"TensorRT engine written to {path}")


if __name__ == "__main__":
    main()
