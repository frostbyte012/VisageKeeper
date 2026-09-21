# VisageKeeper: Denoising Isn't Enough

Certified adversarial robustness for face recognition, with an FPGA-accelerated
smoothed verifier.

> **Paper:** *VisageKeeper: Denoising Isn't Enough* (submitted to DATE 2027)

<!-- ─────────────────────────────────────────────────────────────────────
     TODO: add the system overview figure here
     ![System overview](figures/overview.png)
     ──────────────────────────────────────────────────────────────────── -->

---

## What this is

Purification defenses look strong until the attacker adapts to them. We evaluated
five under one identical adaptive protocol and **every one collapsed to 0.0%**.

VisageKeeper takes a different route: add calibrated Gaussian noise, denoise,
vote over the noisy copies, and return a **certified radius** inside which the
verification decision provably cannot change.

### Headline results (measured, LFW)

| Method | Clean | Adaptive | Proof |
|---|---|---|---|
| No defense | 97.1 | 0.0 | ✗ |
| Feature squeezing | 97.0 | 0.0 | ✗ |
| Noise-injection layer | 96.8 | 0.0 | ✗ |
| Iterative purification (DiffPure-style) | 96.1 | 0.0 | ✗ |
| Ours: denoiser only | 97.0 | 0.0 | ✗ |
| **VisageKeeper** | **91.8** | **25.8** | ✓ |
| *smoothing only (no denoiser)* | *52.7* | *49.5* | ✓ |

Clean = verification accuracy (%). Adaptive = robust TAR (%) under PGD-EOT,
L-inf, eps=8/255, threshold frozen at the clean TAR@FAR=1e-3 point.

**Read both columns together.** The smoothing-only ablation "scores" 49.5 only
because it is already at chance (52.7% clean) and has no accuracy left to lose.
That row is what shows the denoiser is doing the work, not the noise.

### Key findings

1. **Adaptive attacks break purification.** Five defenses, all to 0.0 — including
   our own frequency filter. Fixed attacks (FGSM) badly overstate robustness.
2. **The denoiser makes certification usable.** At sigma=0.50, smoothing alone
   drops face verification to 52.7% (chance). With our DnCNN: 91.8%.
3. **Robustness is one-sided.** The retained 25.8% comes from impersonation
   resistance (ASR 0.485); dodging still succeeds (ASR 1.000). We report this
   rather than quoting a single aggregate.
4. **Certification is affordable.** Alpha-spending stops the vote once the
   majority is settled, with the guarantee unchanged.

<!-- ─────────────────────────────────────────────────────────────────────
     TODO: attenuation-law figure
     ![Attenuation law](figures/attenuation.png)
     ──────────────────────────────────────────────────────────────────── -->

---

## Install

```bash
git clone https://github.com/<you>/VisageKeeper.git
cd VisageKeeper
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. A GPU is optional — everything below runs on CPU (slower).

---

## Get the data

Datasets are **not** in this repo. Download them from the official sources:

### LFW (required — the main evaluation set)

```bash
mkdir -p DATA && cd DATA
wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
tar xzf lfw-funneled.tgz
wget http://vis-www.cs.umass.edu/lfw/pairs.txt
cd ..
```

Then align to 160x160 crops (MTCNN):

```bash
python scripts/prepare_data.py \
  --src DATA/lfw_funneled \
  --dst DATA/lfw_aligned_160 \
  --size 160
```

Expected layout:

```
DATA/
  pairs.txt                 # official 6000-pair protocol
  lfw_aligned_160/
    Aaron_Peirsol/Aaron_Peirsol_0001.png
    ...
```

### CASIA-WebFace (optional — only to retrain denoisers)

Available from the [official page](https://paperswithcode.com/dataset/casia-webface).
Not needed to reproduce any number in the paper; pretrained checkpoints ship in
`checkpoints/`.

---

## Reproduce the results

### 1. Baseline comparison (Table IV)

Each (defense, attack) cell is independent, so run them in parallel:

```bash
for D in none fsqueeze kim_nil iter_purify; do
  for A in clean fgsm pgd_eot bpda_eot; do
    python -m frpure.eval.runner \
      --aligned_dir DATA/lfw_aligned_160 --pairs DATA/pairs.txt \
      --backbone facenet \
      --defenses "$D" --attacks "$A" \
      --iter_denoiser checkpoints/denoiser_s050.pth \
      --n_attack 200 --device cpu \
      --out results/shards/${D}__${A}.csv &
  done
done
wait
```

On a 64-core CPU this takes ~2.5 h wall clock. On a GPU, swap `--device cuda`.

### 2. VisageKeeper: denoising + smoothing (the main result)

```bash
python scripts/eval_smoothed_table4.py \
  --denoiser checkpoints/denoiser_s050.pth --arch dncnn \
  --sigma 0.50 --n_attack 200 --steps 40 \
  --n_eot_score 8 --n_eot_attack 8 \
  --attacks clean,pgd_eot \
  --out results/shards/visagekeeper__smoothed.csv
```

Attacks the **full** chain (noise → denoiser → backbone) by averaging gradients
over the same noise the certificate uses, following Salman et al. (2019). This
is the honest adaptive attack, not a weaker attacker that smoothing trivially
defeats.

### 3. Ablation: smoothing without the denoiser

```bash
python scripts/eval_smoothed_table4.py \
  --denoiser none --sigma 0.50 \
  --n_attack 200 --steps 40 --attacks clean,pgd_eot \
  --out results/shards/ablation_smoothing_only.csv
```

### 4. Build the table

```bash
python scripts/make_table4.py     # -> results/table4_baselines.{csv,tex}
```

### 5. Certification (radius + certified accuracy)

```bash
python scripts/run_certify.py \
  --aligned_dir DATA/lfw_aligned_160 --pairs DATA/pairs.txt \
  --backbone facenet --pretrained vggface2 \
  --denoiser_weights checkpoints/denoiser_s050.pth --arch dncnn \
  --sigma 0.50 --n0 100 --n 1000 --alpha 0.001 --n_pairs 500 \
  --out results/cert_lfw.json
```

Reference output is in `results/cert_arcface_r50_lfw.json`
(96.0% certified at R >= 0.5, mean radius 1.126, ArcFace-R50 backbone).

---

## FPGA (Vitis HLS)

`hls_both/` holds synthesizable C++ for both denoisers, targeting **ZCU104
(xczu7ev-ffvc1156-2-e)**.

```
hls_both/
  dncnn/    denoiser_hls.{cpp,h}      4-conv DnCNN, deployable
  vit/      vit_denoiser_hls.{cpp,h}  ViT denoiser
  run_dncnn_only.tcl / run_vit_only.tcl
  make_table3.py                      parses HLS reports -> Table III
```

Run C-simulation and synthesis:

```bash
vitis_hls -f hls_both/run_dncnn_only.tcl
python hls_both/make_table3.py        # -> CSV/JSON summary
```

**Before synthesizing the ViT**, regenerate its weight blob (excluded here for
size):

```bash
python hls_both/vit/export_vit_weights.py   # -> vit_weights.bin (~6.9 MB)
```

### Notes and caveats

- **ViT does not fit.** At Fixed16 it needs **1441 BRAM (230%)** of the ZCU104's
  624. It is included for completeness; the **DnCNN is the deployable design**.
- **Power is not in Table III.** Vitis HLS does not report power — that requires
  a full Vivado implementation (`synth` → `place & route` → `report_power`).
  Without switching-activity (SAIF) data such numbers are estimates, and should
  be labelled as such.

<!-- ─────────────────────────────────────────────────────────────────────
     TODO: FPGA architecture diagram
     ![Accelerator architecture](figures/accelerator.png)
     ──────────────────────────────────────────────────────────────────── -->

---

## Repository layout

```
frpure/
  attacks/        FGSM, PGD, BPDA, EOT, smoothed-classifier attack
  defenses/       smoothing (SmoothedVerifier), baselines, frpure filter
  eval/           runner.py (matrix), metrics.py, certify
  models/         FaceNet / ArcFace backbone wrappers
  data/           LFW pair loading
scripts/
  eval_smoothed_table4.py   VisageKeeper + ablation evaluation
  make_table4.py            builds Table IV
  prepare_data.py           MTCNN alignment
hls_both/         Vitis HLS sources for both denoisers
section_codes/    figure-generation scripts
checkpoints/      pretrained denoiser + purifier (small)
results/          measured CSV/JSON/TeX from the paper runs
tests/
```

---

## Reproducibility notes

- **Backbones differ between tables.** Table IV (adaptive robustness) uses
  **FaceNet/VGGFace2**; the certification numbers use **ArcFace-R50**. Both are
  stated where reported.
- **Denoiser.** Results use the small 4-conv DnCNN-style residual denoiser
  (`checkpoints/denoiser_s050.pth`, 20,259 params) trained at sigma=0.50 — not
  the 17-layer DnCNN of Zhang et al.
- **`iter_purify` is a DiffPure *analogue*,** not DiffPure. Real DiffPure needs a
  face-trained diffusion model, which we do not have; this reproduces the
  noise-then-denoise *procedure* with our denoiser. Do not cite its numbers as
  Nie et al.'s.
- **Threshold discipline.** The operating threshold is fixed on clean pairs at
  TAR@FAR=1e-3 and never re-tuned on adversarial scores.
- **Randomness.** Smoothing is stochastic; numbers vary by a few tenths of a
  point between runs. Attack subsets use a fixed seed (0).

---

## Citation

```bibtex
@inproceedings{visagekeeper2027,
  title     = {VisageKeeper: Denoising Isn't Enough},
  author    = {TODO},
  booktitle = {Design, Automation and Test in Europe (DATE)},
  year      = {2027}
}
```

### Methods we build on

- Cohen et al., *Certified Adversarial Robustness via Randomized Smoothing*, ICML 2019
- Salman et al., *Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers*, NeurIPS 2019
- Athalye et al., *Obfuscated Gradients Give a False Sense of Security*, ICML 2018
- Xu et al., *Feature Squeezing*, NDSS 2018
- Nie et al., *Diffusion Models for Adversarial Purification*, ICML 2022
- Zhang et al., *Beyond a Gaussian Denoiser (DnCNN)*, IEEE TIP 2017

---

## License

TODO — MIT or Apache-2.0 recommended.
