#!/usr/bin/env python3
import argparse, csv, glob, json, os
KEY_RADII = ["0.25", "0.5", "1.0"]
def parse_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    body = base[len("cert_"):] if base.startswith("cert_") else base
    for d in ["lfw","cfp_fp","cfp_ff","agedb_30","calfw","cplfw"]:
        if body.endswith("_"+d): return body[:-(len(d)+1)], d
    return (body.rsplit("_",1) if "_" in body else (body,"?"))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/cert_*.json")
    ap.add_argument("--out", default="results/master_table.csv")
    a = ap.parse_args(); rows = []
    for path in sorted(glob.glob(a.glob)):
        d = json.load(open(path)); model, ds = parse_name(path)
        ca = {str(k): v for k, v in d.get("certified_accuracy", {}).items()}
        row = {"model": model, "dataset": ds, "sigma": d.get("sigma"), "n": d.get("n"),
               "smoothed_acc": round(d.get("smoothed_accuracy",0),4),
               "abstention": round(d.get("abstention_rate",0),4),
               "mean_R": round(d.get("mean_radius",0),4)}
        for r in KEY_RADII: row[f"cert@{r}"] = round(ca.get(r, ca.get(r+"0", 0.0)), 4)
        rows.append(row)
    if not rows: print("no files matched"); return
    cols = ["model","dataset","sigma","n","smoothed_acc","abstention","mean_R"]+[f"cert@{r}" for r in KEY_RADII]
    w = csv.DictWriter(open(a.out,"w",newline=""), fieldnames=cols); w.writeheader(); w.writerows(rows)
    for r in rows: print(r)
    print("wrote ->", a.out)
if __name__ == "__main__": main()