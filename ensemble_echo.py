#!/usr/bin/env python
"""
Ensemble evaluation of trained ECHO checkpoints (no training).

    python ensemble_echo.py --dataset ICEWS18 --gpu 0 \
        --ckpts checkpoints/ICEWS18_echo_full_s1_best.pt \
                checkpoints/ICEWS18_echo_full_s2_best.pt \
                checkpoints/ICEWS18_echo_full_s3_best.pt

Every checkpoint is evaluated on its own first, so the single-model numbers
can be compared with the ones already reported; a mismatch means the wrong
file was passed. The members are then combined in two ways,

    logprob   mean of the members' log-probabilities (geometric mean)
    prob      mean of the members' probabilities

and BOTH are evaluated on validation and on test. The rule is chosen on
validation MRR; the test number to report is the one of the chosen rule.
Protocol is the one of train_echo.py: time-aware filter, ties averaged.

Whether a checkpoint was trained with the attention across candidates is read
from the checkpoint itself, so files written before and after that option
became the default can be mixed.
"""
import argparse
import dataclasses
import json
import math
import os
import sys

import torch

import train_echo as T
from echo.config import parse_args
from echo.data import EchoData
from echo.model import Echo

RULES = ("logprob", "prob")


def load_members(paths, cfg, data, dev):
    members = []
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=True)["model"]
        compete = any(k.startswith("compete.") for k in sd)
        c = dataclasses.replace(cfg, compete=compete)
        m = Echo(data.num_entities, data.num_relations, c,
                 data.ev, data.t_ptr)
        m.load_state_dict(sd, strict=True)
        members.append((m.to(dev).eval(), c))
        print(f"[member] {os.path.basename(p)}  compete={compete}")
    return members


@torch.no_grad()
def evaluate(members, data, split, dev, cfg, max_snapshots=0):
    K = len(members)
    single = [T.Meters(tuple(cfg.hits_at)) for _ in members]
    ens = {r: T.Meters(tuple(cfg.hits_at)) for r in RULES}
    ds = data.valid_set if split == "valid" else data.test_set

    def filt(lg, raw, it, a, b, objs):
        """time-aware filter, in place; returns the target scores."""
        tgt = lg.gather(1, objs.view(-1, 1))
        rr, oo = data.index.answers(raw["subs"][a:b].numpy(),
                                    raw["rels"][a:b].numpy(), it["t"])
        lg[torch.from_numpy(rr).to(dev), torch.from_numpy(oo).to(dev)] = \
            float("-inf")
        lg.scatter_(1, objs.view(-1, 1), tgt)
        return tgt

    for n, raw in enumerate(T.loader(ds, cfg, False)):
        if max_snapshots and n >= max_snapshots:
            break
        it = T.to_dev(raw, dev)
        ctx = []
        for m, c in members:
            with T.amp(dev):
                ctx.append(T.context(m, it, c))
        for a, b in T.chunks(it["sup_mask"], cfg):
            objs = it["objs"][a:b]
            logp = []
            for (m, c), (E, P, row), M in zip(members, ctx, single):
                with T.amp(dev):
                    lg = T.run_chunk(m, it, E, P, row, a, b).float()
                logp.append(torch.log_softmax(lg, -1))
                tgt = filt(lg, raw, it, a, b, objs)
                M.add("time_aware_filtered", T.ranks_of(lg, tgt))
            stack = torch.stack(logp, 0)                       # (K, B, N)
            comb = dict(logprob=stack.mean(0),
                        prob=torch.logsumexp(stack, 0) - math.log(K))
            del stack, logp
            g = T.strata(it["sup_stat"][a:b], it["sup_mask"][a:b],
                         it["sup_ids"][a:b], objs, it["sup_v"][a:b],
                         {k: it[k] for k in T.EDGE_KEYS})
            for rule, lg in comb.items():
                tgt = filt(lg, raw, it, a, b, objs)
                r = T.ranks_of(lg, tgt)
                ens[rule].add("time_aware_filtered", r)
                for code, nm in enumerate(T.STRATA):
                    if (g == code).any():
                        ens[rule].add(nm, r[g == code])
    return ([M.result()["time_aware_filtered"] for M in single],
            {r: M.result() for r, M in ens.items()})


def line(name, v):
    print(f"  {name:<22} {v['n']:>9,} {v['MRR']*100:>7.2f} "
          f"{v['Hits@1']*100:>7.2f} {v['Hits@3']*100:>7.2f} "
          f"{v['Hits@10']*100:>7.2f}")


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: evaluate only this many snapshots")
    own, rest = ap.parse_known_args()
    cfg = parse_args(rest)
    missing = [p for p in own.ckpts if not os.path.exists(p)]
    if missing:
        sys.exit("checkpoint(s) not found: " + ", ".join(missing))

    dev = T.pick_device(cfg)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        T.AMP = not cfg.no_amp and torch.cuda.is_bf16_supported()
    print(f"ECHO ensemble | dataset={cfg.dataset} members={len(own.ckpts)} "
          f"device={dev}")
    data = EchoData(cfg)
    members = load_members(own.ckpts, cfg, data, dev)

    out = {}
    for split in ("valid", "test"):
        single, ens = evaluate(members, data, split, dev, cfg, own.limit)
        out[split] = dict(single=single, ensemble=ens)
        print(f"\n{split.upper()} -- {cfg.dataset}  (x100, time-aware "
              f"filtered, ties averaged)")
        print(f"  {'':<22} {'n':>9} {'MRR':>7} {'H@1':>7} {'H@3':>7} "
              f"{'H@10':>7}")
        for p, v in zip(own.ckpts, single):
            line(os.path.basename(p)[:22], v)
        mean = {k: sum(v[k] for v in single) / len(single)
                for k in single[0]}
        mean["n"] = single[0]["n"]
        line("mean of members", mean)
        for r in RULES:
            line(f"ensemble [{r}]", ens[r]["time_aware_filtered"])

    best = max(RULES, key=lambda r: out["valid"]["ensemble"][r]
               ["time_aware_filtered"]["MRR"])
    out["selected_rule"] = best
    print(f"\nrule selected on validation MRR: {best}")
    print(f"TEST, ensemble [{best}], by stratum")
    print(f"  {'':<22} {'n':>9} {'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7}")
    res = out["test"]["ensemble"][best]
    for k in ("time_aware_filtered",) + T.STRATA:
        if k in res:
            line(k, res[k])

    path = os.path.join(cfg.save_dir, f"{cfg.dataset}_ensemble_results.json")
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(dataset=cfg.dataset, ckpts=own.ckpts, **out), f,
                  indent=2, default=str)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
