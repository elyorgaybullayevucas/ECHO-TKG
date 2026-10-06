#!/usr/bin/env python
"""
ECHO -- training and evaluation.

    python train_echo.py --dataset ICEWS18 --gpu 0
    python train_echo.py --dataset YAGO    --gpu 1 --seed 2 --tag s2

Ablations: --no_path  --no_type  --no_stream  --no_dyad
           --no_pop  --no_struct   (+ --compete to add candidate attention)

Protocol. Model selection is on validation MRR, time-aware filtered. The test
table reports raw and time-aware filtered, with ties resolved to their
AVERAGE rank (an optimistic tie rule inflates any model that gives many
entities the same score). Test metrics are also split by stratum:

    cold_far   the answer is outside the support and not within two hops
    cold_2hop  outside the support, but a partner of one of the candidates:
               the path intensity can reach it
    dyad_only  s and o interacted, but never under (s, r)
    blocked    (s, r, o) occurred, but a distractor is both more recent and
               more frequent, so no monotone copy score can rank it first
    clean      (s, r, o) occurred and dominates
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from echo.config import parse_args
from echo.data import EchoData, identity_collate
from echo.model import Echo

HDR = (f"{'Ep':>4} | {'Time':>7} | {'Loss':>8} | {'MRR':>7} {'H@1':>7} "
       f"{'H@3':>7} {'H@10':>7} | {'dyadB':>6} {'pathB':>6} {'popB':>6} | "
       f"{'LR':>8}")
STRATA = ("cold_far", "cold_2hop", "dyad_only", "blocked", "clean")
EDGE_KEYS = ("e_dst", "e_rel", "e_dt", "e_cnt", "e_mask")


AMP = False          # set in main(): bf16 autocast when the device supports it


def amp(dev):
    return autocast(dev.type, dtype=torch.bfloat16, enabled=AMP)


def pick_device(cfg):
    """
    --device auto  CUDA when present, otherwise CPU with a loud warning
    --device cuda  CUDA or an error: never fall back silently on a server
    --gpu -1       the GPU with the most free memory right now
    """
    if cfg.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if cfg.device == "cuda":
            raise SystemExit(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Check `nvidia-smi` and that this torch build has CUDA: "
                f"torch {torch.__version__}, built for CUDA "
                f"{torch.version.cuda}.")
        bar = "!" * 70
        print(bar, "!! NO GPU VISIBLE -- running on CPU, which is far too "
              "slow for real runs.", bar, sep="\n", flush=True)
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if cfg.gpu < 0:
        free = [torch.cuda.mem_get_info(i)[0] for i in range(n)]
        cfg.gpu = max(range(n), key=free.__getitem__)
    if cfg.gpu >= n:
        raise SystemExit(f"--gpu {cfg.gpu} but only {n} GPU(s) are visible")
    return torch.device(f"cuda:{cfg.gpu}")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def variant_of(cfg):
    off = [k[3:] for k in ("no_struct", "no_pop", "no_dyad", "no_path",
                           "no_stream", "no_type") if getattr(cfg, k)]
    name = "full" if not off else "no-" + "-".join(off)
    return name + ("+compete" if cfg.compete else "")


def ranks_of(scores, tgt):
    """1 + #better + (#tied - 1) / 2."""
    better = (scores > tgt).sum(1)
    tied = (scores == tgt).sum(1) - 1
    return 1.0 + better.float() + tied.clamp(min=0).float() / 2.0


def strata(stat, mask, ids, objs, sup_v, edges):
    # is the answer the target of an edge leaving one of the candidates?
    hop2 = ((edges["e_dst"][sup_v] == objs.view(-1, 1, 1))
            & edges["e_mask"][sup_v] & mask.unsqueeze(-1)).flatten(1).any(1)
    is_ans = (ids == objs.view(-1, 1)) & mask
    has = stat[..., 3] > 0
    in_sup = is_ans.any(1)
    rec = (is_ans & has).any(1)
    cnt, dt = stat[..., 0], stat[..., 1]
    big = torch.finfo(dt.dtype).max
    a_dt = torch.where(is_ans & has, dt, torch.full_like(dt, big)).min(1).values
    a_ct = torch.where(is_ans & has, cnt, torch.zeros_like(cnt)).max(1).values
    dom = (has & mask & ~is_ans & (dt <= a_dt.unsqueeze(1))
           & (cnt >= a_ct.unsqueeze(1))).any(1)
    out = torch.zeros(objs.numel(), dtype=torch.long, device=objs.device)
    out[~in_sup & hop2] = 1
    out[in_sup & ~rec] = 2
    out[rec & dom] = 3
    out[rec & ~dom] = 4
    return out


class Meters:
    def __init__(self, ks):
        self.k, self.d = ks, {}

    def add(self, name, r):
        s, h, n = self.d.get(name, (0.0, {k: 0.0 for k in self.k}, 0))
        s += (1.0 / r).sum().item()
        for k in self.k:
            h[k] += (r <= k).sum().item()
        self.d[name] = (s, h, n + r.numel())

    def result(self):
        return {nm: dict(MRR=s / n, n=n,
                         **{f"Hits@{k}": h[k] / n for k in self.k})
                for nm, (s, h, n) in self.d.items() if n}


def to_dev(it, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in it.items()}


def loader(ds, cfg, shuffle):
    return DataLoader(ds, batch_size=1, shuffle=shuffle,
                      num_workers=cfg.num_workers,
                      collate_fn=identity_collate,
                      pin_memory=torch.cuda.is_available(),
                      prefetch_factor=4 if cfg.num_workers > 0 else None)


def chunks(mask, cfg):
    """Query ranges holding at most query_chunk queries and cand_budget
    (query, candidate) pairs."""
    c = mask.sum(1).cumsum(0).tolist()
    n, a, out = len(c), 0, []
    while a < n:
        base = c[a - 1] if a else 0
        b = a + 1
        while (b < n and b - a < cfg.query_chunk
               and c[b] - base <= cfg.cand_budget):
            b += 1
        out.append((a, b))
        a = b
    return out


def context(model, it, cfg):
    """Everything that depends on the timestamp but not on the query."""
    E = model.evolve(it["t"])
    if cfg.no_pop:
        return E, None, None
    rel_u, row = torch.unique(it["rels"], return_inverse=True)
    return E, model.popularity(it["t"], rel_u), row


def run_chunk(model, it, E, P, row, a, b, return_struct=False):
    return model(E, None if P is None else P[row[a:b]],
                 it["subs"][a:b], it["rels"][a:b], it["sup_ids"][a:b],
                 it["sup_stat"][a:b], it["sup_mask"][a:b],
                 it["tok_r"][a:b], it["tok_d"][a:b],
                 sup_v=it["sup_v"][a:b],
                 edges={k: it[k] for k in EDGE_KEYS},
                 return_struct=return_struct)


@torch.no_grad()
def evaluate(model, data, split, dev, cfg, stratify=False):
    model.eval()
    M = Meters(tuple(cfg.hits_at))
    ds = data.valid_set if split == "valid" else data.test_set
    for raw in loader(ds, cfg, False):
        it = to_dev(raw, dev)
        with amp(dev):
            E, P, row = context(model, it, cfg)
        for a, b in chunks(it["sup_mask"], cfg):
            with amp(dev):
                lg = run_chunk(model, it, E, P, row, a, b).float()
            objs = it["objs"][a:b]
            tgt = lg.gather(1, objs.view(-1, 1))
            M.add("raw", ranks_of(lg, tgt))
            rr, oo = data.index.answers(raw["subs"][a:b].numpy(),
                                        raw["rels"][a:b].numpy(), it["t"])
            lg[torch.from_numpy(rr).to(dev), torch.from_numpy(oo).to(dev)] = \
                float("-inf")
            lg.scatter_(1, objs.view(-1, 1), tgt)
            r = ranks_of(lg, tgt)
            M.add("time_aware_filtered", r)
            if stratify:
                g = strata(it["sup_stat"][a:b], it["sup_mask"][a:b],
                           it["sup_ids"][a:b], objs, it["sup_v"][a:b],
                           {k: it[k] for k in EDGE_KEYS})
                for code, nm in enumerate(STRATA):
                    if (g == code).any():
                        M.add(nm, r[g == code])
    return M.result()


def main():
    cfg = parse_args()
    variant = variant_of(cfg)
    set_seed(cfg.seed)
    global AMP
    dev = pick_device(cfg)
    cuda = dev.type == "cuda"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        AMP = not cfg.no_amp and torch.cuda.is_bf16_supported()
        free, tot = torch.cuda.mem_get_info(dev)
        print(f"[GPU {cfg.gpu}] {torch.cuda.get_device_name(dev)}  "
              f"{free/2**30:.1f}/{tot/2**30:.1f} GB free  "
              f"bf16 autocast={'on' if AMP else 'off'}")
    elif cfg.force_amp:
        AMP = True                      # exercises the autocast path on CPU

    print(f"ECHO | dataset={cfg.dataset} variant={variant} "
          f"tag={cfg.tag or '-'} seed={cfg.seed} device={dev}")
    print(f"  d={cfg.embed_dim} H={cfg.hist_len} stream: L={cfg.stream_len} "
          f"dim={cfg.stream_dim} layers={cfg.stream_layers} | support: "
          f"dyad={cfg.dyad_support} triple={cfg.triple_support} "
          f"path={cfg.path_support}")

    data = EchoData(cfg)
    model = Echo(data.num_entities, data.num_relations, cfg,
                 data.ev, data.t_ptr).to(dev)
    print(f"[model] params="
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optim = AdamW(model.parameters(), lr=cfg.lr,
                  weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.epochs * cfg.warmup_ratio))
    sched = SequentialLR(optim, [
        LinearLR(optim, 0.1, 1.0, total_iters=warm),
        CosineAnnealingLR(optim, T_max=max(1, cfg.epochs - warm),
                          eta_min=cfg.lr * 0.02)], milestones=[warm])

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    name = f"{cfg.dataset}_echo_{variant}" + (f"_{cfg.tag}" if cfg.tag else "")
    ck = os.path.join(cfg.save_dir, f"{name}_best.pt")
    log_path = os.path.join(cfg.log_dir, f"{name}.jsonl")
    best, best_ep, bad = 0.0, 0, 0
    print("\n" + HDR + "\n" + "-" * len(HDR))

    for ep in ([] if cfg.eval_only else range(1, cfg.epochs + 1)):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for k, raw in enumerate(loader(data.train_set, cfg, True)):
            if cfg.max_snapshots and k >= cfg.max_snapshots:
                break
            it = to_dev(raw, dev)
            optim.zero_grad(set_to_none=True)

            # The context is computed once per timestamp and the graph is cut
            # there. Each chunk then frees its own graph on backward while
            # dL/dE and dL/dP accumulate in the detached copies; one more
            # backward carries them through the evolver and the popularity
            # head. Same gradient, memory bounded by a single chunk.
            with amp(dev):
                E, P, row = context(model, it, cfg)
            E_d = E.detach().requires_grad_(True)
            P_d = None if P is None else P.detach().requires_grad_(True)

            n, loss_t = it["subs"].numel(), 0.0
            for a, b in chunks(it["sup_mask"], cfg):
                with amp(dev):
                    lg, ls = run_chunk(model, it, E_d, P_d, row, a, b, True)
                    obj = it["objs"][a:b]
                    lc = F.cross_entropy(lg, obj,
                                         label_smoothing=cfg.label_smoothing)
                    if cfg.struct_aux > 0 and not cfg.no_struct:
                        lc = lc + cfg.struct_aux * F.cross_entropy(
                            ls, obj, label_smoothing=cfg.label_smoothing)
                w = (b - a) / n
                (lc * w).backward()
                loss_t += lc.item() * w

            tail = 0.0
            if E.requires_grad and E_d.grad is not None:
                tail = tail + (E * E_d.grad).sum()
            if P is not None and P_d.grad is not None:
                tail = tail + (P * P_d.grad).sum()
            if torch.is_tensor(tail):
                tail.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            tot += loss_t; nb += 1
        sched.step()

        m = evaluate(model, data, "valid", dev, cfg)["time_aware_filtered"]
        is_best = m["MRR"] > best
        if is_best:
            best, best_ep, bad = m["MRR"], ep, 0
            torch.save({"model": model.state_dict()}, ck)
        else:
            bad += 1
        print(f"{'*' if is_best else ' '}{ep:>3} | {time.time()-t0:>6.1f}s | "
              f"{tot/max(nb,1):>8.4f} | {m['MRR']:>7.4f} {m['Hits@1']:>7.4f} "
              f"{m['Hits@3']:>7.4f} {m['Hits@10']:>7.4f} | "
              f"{model.dyad_bias.item():>6.2f} "
              f"{model.path_bias.item():>6.2f} "
              f"{model.pop_bias.item():>6.2f} | "
              f"{sched.get_last_lr()[0]:>8.2e}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": ep, "loss": tot / max(nb, 1),
                                "valid": m}) + "\n")
        if bad >= cfg.patience:
            print(f"[early-stop] {cfg.patience} epochs without improvement")
            break

    print(f"\nbest valid MRR = {best:.4f} (epoch {best_ep})")
    if os.path.exists(ck):
        model.load_state_dict(
            torch.load(ck, map_location=dev, weights_only=True)["model"])
    res = evaluate(model, data, "test", dev, cfg, stratify=True)

    print(f"\nTEST -- {cfg.dataset} [{variant}]  (x100, ties averaged)")
    print(f"  {'':<22} {'n':>9} {'MRR':>7} {'H@1':>7} {'H@3':>7} {'H@10':>7}")
    for k in ("raw", "time_aware_filtered") + STRATA:
        if k in res:
            v = res[k]
            print(f"  {k:<22} {v['n']:>9,} {v['MRR']*100:>7.2f} "
                  f"{v['Hits@1']*100:>7.2f} {v['Hits@3']*100:>7.2f} "
                  f"{v['Hits@10']*100:>7.2f}")
    out = os.path.join(cfg.save_dir, f"{name}_results.json")
    with open(out, "w") as f:
        json.dump(dict(dataset=cfg.dataset, variant=variant, tag=cfg.tag,
                       seed=cfg.seed, best_epoch=best_ep, valid_mrr=best,
                       test=res, config=vars(cfg)), f, indent=2, default=str)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
