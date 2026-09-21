#!/usr/bin/env python3
"""VisageKeeper (denoise + Gaussian smoothing/voting) under the SAME protocol
as the Table IV baselines.

Reports, at the clean TAR@FAR=1e-3 operating threshold:
  * clean      -- no attack
  * pgd_eot    -- adaptive PGD+EOT through the FULL smoothed chain
                  (noise -> denoiser -> backbone), i.e. Salman et al. 2019.

The smoothed score is the SAME quantity the certificate is built on
(E_eps[cos]), so this is the empirical counterpart to certification, not a
different defense.
"""
import argparse, csv, json, os, numpy as np, torch

from frpure.data.lfw import LFWPairs, sample_attack_subset
from frpure.models.backbone import build_backbone
from frpure.defenses.smoothing import build_denoiser, SmoothedVerifier
from frpure.attacks.smoothed_attack import pgd_eot_smoothed
from frpure.eval import metrics as M
from frpure.attacks import gradient as G


def smoothed_scores(sv, probes, galleries, n_eot, batch=32):
    """E_eps[cos] for each pair -- no grad, used for scoring/metrics."""
    out = []
    with torch.no_grad():
        for i in range(0, len(probes), batch):
            p = probes[i:i+batch].to(sv.device)
            g = galleries[i:i+batch].to(sv.device)
            ge = sv.backbone.embed(g)
            acc = 0
            for _ in range(n_eot):
                x = (p + torch.randn_like(p) * sv.sigma).clamp(0, 1)
                if sv.denoiser is not None:
                    x = sv.denoiser(x)
                acc = acc + (sv.backbone.embed(x) * ge).sum(1)
            out.append((acc / n_eot).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", default="DATA/lfw_aligned_160")
    ap.add_argument("--pairs", default="DATA/pairs.txt")
    ap.add_argument("--denoiser", default="results/denoiser_s050.pth")
    ap.add_argument("--arch", default="dncnn")
    ap.add_argument("--sigma", type=float, default=0.50)
    ap.add_argument("--n_attack", type=int, default=200)
    ap.add_argument("--n_eot_score", type=int, default=8)
    ap.add_argument("--n_eot_attack", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--attacks", default="clean,pgd_eot")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/shards/visagekeeper__smoothed.csv")
    a = ap.parse_args()

    dev = a.device
    bb = build_backbone("facenet", device=dev)
    if a.denoiser.lower() in ("none", ""):
        # ABLATION: Gaussian smoothing WITHOUT the denoiser. Isolates how much
        # of the robustness comes from smoothing alone (Cohen et al. 2019) vs
        # from our trained denoiser.
        den = None
        print("ABLATION: smoothing with NO denoiser", flush=True)
    else:
        den = build_denoiser(a.arch).to(dev).eval()
        sd = torch.load(a.denoiser, map_location=dev)
        if any(k.startswith("core.") for k in sd):
            sd = {k[5:]: v for k, v in sd.items() if k.startswith("core.")}
        den.load_state_dict(sd)
        for p in den.parameters():
            p.requires_grad_(False)
    sv = SmoothedVerifier(bb, denoiser=den, sigma=a.sigma, device=dev)

    ds = LFWPairs(a.aligned_dir, a.pairs)
    all1 = torch.stack([ds[i].img1 for i in range(len(ds))])
    all2 = torch.stack([ds[i].img2 for i in range(len(ds))])
    same = np.array([ds.pairs[i][2] for i in range(len(ds))])

    # clean threshold on the SMOOTHED score (same protocol as baselines)
    print(f"scoring {len(ds)} clean pairs (sigma={a.sigma}, n_eot={a.n_eot_score}) ...",
          flush=True)
    s_clean = smoothed_scores(sv, all1, all2, a.n_eot_score)
    gen, imp = s_clean[same], s_clean[~same]
    ver_acc, _ = M.verification_accuracy(gen, imp)
    taf = M.tar_at_far(gen, imp, (1e-2, 1e-3))
    op_thr = taf[1e-3]["threshold"]
    print(f"  ver_acc={ver_acc:.4f}  op_thr={op_thr:.4f}", flush=True)

    sub = sample_attack_subset(ds, a.n_attack, a.n_attack)
    di, ii = sub["dodging"], sub["impersonation"]
    rows = []

    for atk in a.attacks.split(","):
        if atk == "clean":
            rt = M.robust_tar(gen, imp, op_thr)
            rows.append(dict(defense="visagekeeper_smoothed", attack="clean",
                             ver_acc=ver_acc, robust_tar=rt,
                             asr_dodging="", asr_impersonation="",
                             sigma=a.sigma, tar_at_far=json.dumps(
                                 {str(k): v for k, v in taf.items()})))
            print(f"[visagekeeper | clean] ver_acc={ver_acc:.3f} robust_tar={rt:.3f}",
                  flush=True)
            continue

        if atk != "pgd_eot":
            raise SystemExit(f"unsupported attack {atk}")

        eps, alpha = G.DEFAULT_EPS_LINF, G.DEFAULT_ALPHA_LINF
        adv_scores = {}
        for mode, idxs in (("dodging", di), ("impersonation", ii)):
            outs = []
            for n, k in enumerate(idxs):
                probe, gal = all1[k].to(dev), all2[k].to(dev)
                with torch.no_grad():
                    ge = bb.embed(gal.unsqueeze(0)).squeeze(0)
                adv = pgd_eot_smoothed(sv, probe, ge, eps, alpha, a.steps,
                                       mode, n_eot=a.n_eot_attack)
                outs.append(adv.detach().cpu())
                if (n + 1) % 20 == 0:
                    print(f"  {mode}: {n+1}/{len(idxs)}", flush=True)
            advs = torch.stack(outs)
            gals = all2[idxs]
            adv_scores[mode] = smoothed_scores(sv, advs, gals, a.n_eot_score)

        asr_d = M.attack_success_rate(adv_scores["dodging"], op_thr, "dodging")
        asr_i = M.attack_success_rate(adv_scores["impersonation"], op_thr,
                                      "impersonation")
        rt = M.robust_tar(adv_scores["dodging"], adv_scores["impersonation"], op_thr)
        rows.append(dict(defense="visagekeeper_smoothed", attack="pgd_eot",
                         ver_acc=ver_acc, robust_tar=rt, asr_dodging=asr_d,
                         asr_impersonation=asr_i, sigma=a.sigma,
                         tar_at_far=json.dumps({str(k): v for k, v in taf.items()})))
        print(f"[visagekeeper | pgd_eot] robust_tar={rt:.3f} "
              f"asr_dodge={asr_d:.3f} asr_imp={asr_i:.3f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
