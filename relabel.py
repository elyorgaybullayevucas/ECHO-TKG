#!/usr/bin/env python
"""
One-off: competition became the default. Results written while it was
opt-in carry variant "full+compete" (now "full") or "full" (now
"no-compete"). Rewrites the variant field in place; nothing else changes.
"""
import glob, json
for p in glob.glob("checkpoints/*_results.json"):
    r = json.load(open(p))
    v = r["variant"]
    new = {"full+compete": "full", "full": "no-compete"}.get(v)
    if new:
        r["variant"] = new
        json.dump(r, open(p, "w"), indent=2, default=str)
        print(f"{p}: {v} -> {new}")
