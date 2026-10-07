#!/usr/bin/env python
"""
One-off: competition became the default. Results written while it was
opt-in carry variant "full+compete" (now "full") or "full" (now
"no-compete"). Rewrites the variant field AND renames the file so that new
runs cannot overwrite the old ones.
"""
import glob, json, os
ren = {"full+compete": "full", "full": "no-compete"}
# rename "full" first would collide with "full+compete" -> "full"; so handle
# the "full" files first, then the "full+compete" files.
for old, new in (("full", "no-compete"), ("full+compete", "full")):
    for p in sorted(glob.glob("checkpoints/*_results.json")):
        r = json.load(open(p))
        if r["variant"] != old:
            continue
        r["variant"] = new
        q = p.replace(f"_echo_{old}_", f"_echo_{new}_")
        if q == p:
            q = p.replace("_results.json", f"_{new}_results.json")
        assert not os.path.exists(q), q
        json.dump(r, open(q, "w"), indent=2, default=str)
        os.remove(p)
        print(f"{os.path.basename(p)} -> {os.path.basename(q)}  [{old} -> {new}]")
