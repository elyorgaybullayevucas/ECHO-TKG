#!/usr/bin/env python
"""
Every finished run, one block per (dataset, variant, recipe), best
validation MRR first. Runs that differ only in seed are pooled: a tag like
"s2" or "e16_s2" names seed 2 of recipe "" or "e16".
"""
import glob, json, os, re, statistics as st
from collections import defaultdict

runs = defaultdict(list)
for p in sorted(glob.glob(os.path.join("checkpoints", "*_results.json"))):
    r = json.load(open(p))
    recipe = re.sub(r"_?s\d+$", "", r.get("tag") or "")
    runs[(r["dataset"], r["variant"], recipe)].append(r)

rows = ("time_aware_filtered", "cold_far", "cold_2hop", "cold", "dyad_only",
        "blocked", "clean")
valid = lambda rs: st.mean(100 * r["valid_mrr"] for r in rs)
for (ds, var, recipe), rs in sorted(runs.items(),
                                    key=lambda kv: (kv[0][0], -valid(kv[1]))):
    print(f"\n{ds} [{var}] recipe={recipe or '-'}  "
          f"seeds={sorted(r['seed'] for r in rs)}  "
          f"valid MRR {valid(rs):.2f}  "
          f"best epoch {[r['best_epoch'] for r in rs]}")
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
