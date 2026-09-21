#!/usr/bin/env python3
"""Build Table III from the Vitis HLS csynth reports of BOTH denoisers.

    python make_table3.py

Reads the "Available" row from each report rather than hardcoding the device,
so the percentages stay correct if you retarget the part.
"""
import csv, glob, json, re, sys

NAME = {32: "Float32", 16: "Fixed16", 8: "Fixed8"}
FALLBACK_AVAIL = [624, 1728, 460800, 230400]   # ZCU104 XCZU7EV


def parse(path):
    t = open(path, errors="ignore").read()
    out = {}
    # Latency: take the FIRST data row of the "+ Latency: * Summary" table.
    # Anchor on the "* Summary:" that follows "+ Latency:", and stop at the
    # next "+ Detail:" AFTER that point -- searching for "+ Detail:" from the
    # start of the file lands in the Timing section and slices the wrong span.
    li = t.find("+ Latency:")
    lat = ""
    if li >= 0:
        lj = t.find("+ Detail:", li)
        lat = t[li:lj] if lj > li else t[li:li + 2000]
    # rows look like:  |  743429641|  743429641|  7.434 sec|  7.434 sec| ...
    m = re.search(r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s+(ns|us|ms|sec)\s*\|', lat)
    if m:
        out["cycles"] = int(m.group(2))
        v, u = float(m.group(3)), m.group(4)
        out["ms"] = v * {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "sec": 1e3}[u]
    else:
        m = re.search(r'\|\s*(\d{3,})\s*\|\s*(\d{3,})\s*\|', lat)
        if m:
            out["cycles"] = int(m.group(2))
            out["ms"] = int(m.group(2)) / 1e5          # cycles @100 MHz

    # IMPORTANT: take the Total/Available rows from the "Utilization
    # Estimates -> Summary" block only. Later tables (Instance, Memory) also
    # contain a "Total" row, and matching those yields submodule numbers.
    seg = t
    i = t.find("Utilization Estimates")
    if i >= 0:
        j = t.find("+ Detail:", i)
        seg = t[i:j] if j > i else t[i:]
    for key, pat in (("tot", r'\|\s*Total\s*\|'), ("avail", r'\|\s*Available\s*\|')):
        m = re.search(pat + r'\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|', seg)
        if m:
            out[key] = [int(m.group(i)) for i in (1, 2, 3, 4)]
    return out


def collect(tag, globs):
    rows = []
    for prec in (32, 16, 8):
        hits = []
        for g in globs:
            hits += glob.glob(g.format(p=prec), recursive=True)
        if not hits:
            print(f"  [skip] {tag} PREC={prec}: no report", file=sys.stderr)
            continue
        r = parse(hits[0])
        if "tot" not in r:
            print(f"  [warn] {tag} PREC={prec}: unparsed {hits[0]}", file=sys.stderr)
            continue
        r["prec"] = prec
        rows.append(r)
    return rows


def emit(label, rows):
    if not rows:
        return
    print(f"\\multicolumn{{7}}{{l}}{{\\textit{{{label}}}}} \\\\")
    for r in rows:
        avail = r.get("avail") or FALLBACK_AVAIL
        cells = [f"{v} ({100.0*v/a:.1f}\\%)" for v, a in zip(r["tot"], avail)]
        if "ms" in r:
            ms = r["ms"]
            lat = f"{ms:.0f}" if ms >= 10 else f"{ms:.2f}"
        else:
            lat = "---"
        print(f"\\textit{{VisageKeeper}} ({NAME[r['prec']]}) & "
              + " & ".join(cells) + f" & {lat} & --- \\\\")


dn = collect("dncnn", ["dncnn/dncnn_{p}/sol/syn/report/*_csynth.rpt",
                       "dncnn_{p}/sol/syn/report/*_csynth.rpt",
                       "**/dncnn_{p}/**/*_csynth.rpt"])
vt = collect("vit",   ["vit/vit_{p}/sol/syn/report/*_csynth.rpt",
                       "vit_{p}/sol/syn/report/*_csynth.rpt",
                       "**/vit_{p}/**/*_csynth.rpt"])

if not dn and not vt:
    sys.exit("No reports found. Has csynth finished?")

print("\n% ===== paste into Table III =====")
emit("DnCNN denoiser (deployed at $\\sigma \\lesssim 0.5$)", dn)
if dn and vt:
    print("\\midrule")
emit("ViT denoiser (deployed at $\\sigma \\gtrsim 0.5$)", vt)
print("\n% Columns: BRAM_18K, DSP, FF, LUT, latency(ms @100MHz), power(W)")
print("% Power needs Vivado: open the exported IP, implement, Report Power.")

for tag, rows in (("DnCNN", dn), ("ViT", vt)):
    for r in rows:
        avail = r.get("avail") or FALLBACK_AVAIL
        if r["tot"][0] / avail[0] > 0.9:
            print(f"\n[!] {tag} {NAME[r['prec']]}: BRAM {100*r['tot'][0]/avail[0]:.0f}%"
                  " -- lower PF_TOK/PF_W (ViT) before trusting place-and-route.")

# ---------------------------------------------------------------- exports
rows = []
for tag, rs in (("DnCNN", dn), ("ViT", vt)):
    for r in rs:
        avail = r.get("avail") or FALLBACK_AVAIL
        rows.append({
            "denoiser":  tag,
            "precision": NAME[r["prec"]],
            "BRAM_18K":  r["tot"][0], "BRAM_pct":  round(100.0*r["tot"][0]/avail[0], 1),
            "DSP":       r["tot"][1], "DSP_pct":   round(100.0*r["tot"][1]/avail[1], 1),
            "FF":        r["tot"][2], "FF_pct":    round(100.0*r["tot"][2]/avail[2], 1),
            "LUT":       r["tot"][3], "LUT_pct":   round(100.0*r["tot"][3]/avail[3], 1),
            "cycles":    r.get("cycles"),
            "latency_ms": round(r["ms"], 3) if "ms" in r else None,
            "power_W":   None,
        })

with open("table3.csv", "w", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
with open("table3.json", "w") as f:
    json.dump({"part": "xczu7ev-ffvc1156-2-e", "clock_MHz": 100,
               "available": {"BRAM_18K": FALLBACK_AVAIL[0], "DSP": FALLBACK_AVAIL[1],
                             "FF": FALLBACK_AVAIL[2], "LUT": FALLBACK_AVAIL[3]},
               "rows": rows}, f, indent=2)

print("\n% wrote table3.csv and table3.json  <- paste either of these")
print("\n=== table3.csv ===")
print(open("table3.csv").read())
