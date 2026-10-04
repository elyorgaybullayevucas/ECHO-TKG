#!/usr/bin/env python
"""Mean and standard deviation over seeds of every finished run."""
import glob, json, os, statistics as st
from collections import defaultdict

runs = defaultdict(list)
for p in sorted(glob.glob(os.path.join("checkpoints", "*_results.json"))):
    r = json.load(open(p))
    runs[(r["dataset"], r["variant"])].append(r)
rows = ("time_aware_filtered", "cold", "dyad_only", "blocked", "clean")
for (ds, var), rs in sorted(runs.items()):
    print(f"\n{ds} [{var}]  seeds={sorted(r['seed'] for r in rs)}")
    print(f"  {'':<22} {'MRR':>13} {'H@1':>13} {'H@3':>13} {'H@10':>13}")
    for k in rows:
        if not all(k in r["test"] for r in rs):
            continue
        cells = []
        for m in ("MRR", "Hits@1", "Hits@3", "Hits@10"):
            v = [100 * r["test"][k][m] for r in rs]
            sd = st.stdev(v) if len(v) > 1 else 0.0
            cells.append(f"{st.mean(v):>6.2f} +-{sd:>4.2f}")
        print(f"  {k:<22} " + " ".join(f"{c:>13}" for c in cells))
