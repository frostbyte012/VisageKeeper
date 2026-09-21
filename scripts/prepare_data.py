#!/usr/bin/env python3
"""
scripts/prepare_data.py -- fetch LFW (funneled) directly from figshare with a
browser User-Agent (sklearn's downloader sends none and gets 403'd), extract,
MTCNN-align, and write a single aligned folder used for BOTH train and eval:

  DATA/lfw_aligned_<size>/<Name>/<Name>_0001.png
  DATA/pairs.txt                                  (official 6000-pair benchmark)

figshare file IDs (confirmed from sklearn + your retry log):
  5976015 = lfw-funneled.tgz   |   5976006 = pairs.txt

If figshare is ever blocked, download those two on any machine and drop them in
DATA/ (download is skipped when the files already exist):
  DATA/lfw-funneled.tgz   <- https://ndownloader.figshare.com/files/5976015
  DATA/pairs.txt          <- https://ndownloader.figshare.com/files/5976006
"""
import argparse
import os
import shutil
import sys
import tarfile
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FUNNELED_URL = "https://ndownloader.figshare.com/files/5976015"   # lfw-funneled.tgz
PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"      # pairs.txt (6000)
UA = "Mozilla/5.0"


def download(url, dst, retries=4):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  exists, skip: {dst}")
        return
    import time
    for attempt in range(1, retries + 1):
        try:
            print(f"  downloading {url}  ->  {dst}  (try {attempt})")
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
            return
        except Exception as e:
            print(f"    failed: {e}")
            if os.path.exists(dst):
                os.remove(dst)
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"could not download {url} after {retries} tries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=160, choices=[112, 160])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tgz = os.path.join(args.out, "lfw-funneled.tgz")
    pairs_dst = os.path.join(args.out, "pairs.txt")
    aligned = os.path.join(args.out, f"lfw_aligned_{args.size}")

    try:
        download(FUNNELED_URL, tgz)
        download(PAIRS_URL, pairs_dst)
    except Exception as e:
        print(f"\nDOWNLOAD FAILED: {e}\nPlace these in {args.out} manually and re-run:")
        print(f"  lfw-funneled.tgz <- {FUNNELED_URL}\n  pairs.txt        <- {PAIRS_URL}")
        sys.exit(1)

    # locate / create the extracted raw dir
    raw_dir = os.path.join(args.out, "lfw_funneled")
    if not os.path.isdir(raw_dir):
        print("  extracting lfw-funneled.tgz ...")
        with tarfile.open(tgz) as t:
            t.extractall(args.out)
    if not os.path.isdir(raw_dir):
        cands = [d for d in os.listdir(args.out)
                 if d.lower().startswith("lfw") and "aligned" not in d
                 and os.path.isdir(os.path.join(args.out, d))]
        raw_dir = os.path.join(args.out, sorted(cands)[0])

    # align every image with MTCNN (fallback: center-resize if no face found)
    import torch
    from PIL import Image
    from facenet_pytorch import MTCNN
    dev = args.device if torch.cuda.is_available() else "cpu"
    mt = MTCNN(image_size=args.size, margin=0, post_process=False,
               select_largest=True, device=dev)

    os.makedirs(aligned, exist_ok=True)
    people = sorted(d for d in os.listdir(raw_dir)
                    if os.path.isdir(os.path.join(raw_dir, d)))
    nok = nfb = 0
    for k, person in enumerate(people):
        pin = os.path.join(raw_dir, person)
        pout = os.path.join(aligned, person)
        os.makedirs(pout, exist_ok=True)
        for fn in sorted(os.listdir(pin)):
            stem = os.path.splitext(fn)[0]
            outp = os.path.join(pout, stem + ".png")
            if os.path.exists(outp):
                nok += 1
                continue
            img = Image.open(os.path.join(pin, fn)).convert("RGB")
            face = mt(img)
            if face is None:
                arr = np.asarray(img.resize((args.size, args.size)), dtype="uint8")
                nfb += 1
            else:
                a = (face.clamp(0, 255) / 255.0).permute(1, 2, 0).cpu().numpy()
                arr = (a * 255).astype("uint8")
            Image.fromarray(arr).save(outp)
            nok += 1
        if (k + 1) % 400 == 0:
            print(f"  aligned {k+1}/{len(people)} people (ok={nok} fallback={nfb})")

    print(f"\nDONE. aligned -> {aligned} (ok={nok} fallback={nfb})\npairs -> {pairs_dst}")
    print("\nNext:")
    print(f"  python scripts/train_purifier.py --aligned_dir {aligned} "
          f"--backbone facenet --size {args.size} --epochs 5 --out results/purifier.pth")
    print(f"  python -m frpure.eval.runner --aligned_dir {aligned} --pairs {pairs_dst} "
          f"--backbone facenet --frpure_weights results/purifier.pth --out results/lfw_trained.csv")


if __name__ == "__main__":
    main()