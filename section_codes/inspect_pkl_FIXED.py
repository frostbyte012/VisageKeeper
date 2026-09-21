"""
inspect_pkl.py — reveals exact key names and shapes inside VGGFace2 .pkl files
Usage:  python inspect_pkl.py --weights weights/
"""
import pickle, sys, argparse
import numpy as np
from pathlib import Path

def inspect(pkl_path):
    print(f"\n{'='*70}")
    print(f"FILE: {Path(pkl_path).name}  ({Path(pkl_path).stat().st_size/1e6:.1f} MB)")
    print(f"{'='*70}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    print(f"Top-level type: {type(data)}")

    # Unwrap common wrappers
    if isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())[:8]}")
        if "state_dict" in data:
            data = data["state_dict"]
            print("  → unwrapped 'state_dict'")
        elif "model" in data:
            data = data["model"]
            print("  → unwrapped 'model'")

    # Strip DataParallel 'module.' prefix
    items = {}
    for k, v in data.items():
        key = k[7:] if k.startswith("module.") else k
        items[key] = v

    total = len(items)
    print(f"\nTotal tensors: {total}")

    def shape_of(v):
        if isinstance(v, np.ndarray): return v.shape
        if hasattr(v, "shape"):       return tuple(v.shape)
        return type(v).__name__

    print(f"\n── FIRST 25 keys ──")
    for k in list(items)[:25]:
        print(f"  {k:<60} {str(shape_of(items[k]))}")

    print(f"\n── LAST 10 keys ──")
    for k in list(items)[-10:]:
        print(f"  {k:<60} {str(shape_of(items[k]))}")

    # Architecture hints
    print(f"\n── ARCHITECTURE HINTS ──")
    top_prefixes = sorted({k.split(".")[0] for k in items})
    print(f"  Top-level module names : {top_prefixes}")

    has_se      = any("se" in k.lower() for k in items)
    has_layer   = any("layer" in k for k in items)
    has_fc      = any(k.startswith("fc") for k in items)
    has_last_fc = any("last_linear" in k or "logits" in k for k in items)
    print(f"  Has SE blocks          : {has_se}")
    print(f"  Has torchvision layers : {has_layer}")
    print(f"  Has 'fc' key           : {has_fc}")
    print(f"  Has last linear/logits : {has_last_fc}")

    # Find the likely output / embedding layer
    output_keys = [k for k in items if any(
        x in k for x in ["fc", "classifier", "last_linear", "output", "logit"])]
    print(f"\n── OUTPUT / FC LAYER KEYS ──")
    for k in output_keys:
        print(f"  {k:<60} {str(shape_of(items[k]))}")

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="weights/")
args = parser.parse_args()

for pkl in sorted(Path(args.weights).glob("*.pkl")):
    inspect(str(pkl))

print("\nDone.")