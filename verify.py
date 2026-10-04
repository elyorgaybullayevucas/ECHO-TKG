#!/usr/bin/env python
"""
Two checks that must pass before any number from this code is believed.

1. NO LEAKAGE. Supports, streams, snapshot graphs and the popularity field
   at time t are recomputed after every event at or after t has been deleted
   from the data. They must be identical: nothing at t or later is read.

2. THE VECTORISED SUPPORT IS RIGHT. Candidates, statistics and streams are
   compared with a brute-force loop over the raw events.

    python verify.py --dataset ICEWS14s
"""
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from echo.config import DATASETS, EchoConfig
from echo.data import EchoData, EchoIndex, build_support
from echo.model import Echo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ICEWS14s")
    ap.add_argument("--data_dir", default="./data")
    a = ap.parse_args()
    cfg = EchoConfig(**dict(DATASETS[a.dataset], dataset=a.dataset,
                            data_dir=a.data_dir))
    D = EchoData(cfg)
    ix, N, R2 = D.index, D.index.N, D.index.R2
    S1, S2, L = cfg.dyad_support, cfg.triple_support, cfg.stream_len
    ds = D.test_set
    i = len(ds) // 2
    t = int(ds.times[i])
    blk = ds.q[ds.starts[i]:ds.ends[i]]
    subs, rels = blk[:, 0], blk[:, 1]

    # ── 1. leakage ───────────────────────────────────────────────────────────
    full = build_support(ix, t, subs, rels, S1, S2, L)
    past = D.ev[D.ev[:, 3] < t]
    cut = build_support(EchoIndex(past, N, R2, D.T), t, subs, rels, S1, S2, L)
    for name, x, y in zip(("ids", "stat", "mask", "tok_r", "tok_d"), full, cut):
        assert np.array_equal(x, y), f"support leaks through {name}"

    m_full = Echo(N, D.num_relations, cfg, D.ev, D.t_ptr)
    tp = np.searchsorted(past[:, 3], np.arange(D.T + 1))
    m_cut = Echo(N, D.num_relations, cfg, past, tp)
    ru = torch.unique(torch.from_numpy(rels))
    for x, y in zip(m_full.pop_features(t, ru), m_cut.pop_features(t, ru)):
        assert torch.equal(x, y), "popularity field leaks"
    for g, h in zip(m_full.history(t), m_cut.history(t)):
        assert all(torch.equal(x, y) for x, y in zip(g, h)), "graph leaks"
    print(f"[ok] no leakage: t={t}, {len(subs)} queries, "
          f"{len(D.ev) - len(past):,} future events removed, nothing changed")

    # ── 2. brute force ───────────────────────────────────────────────────────
    ids, stat, mask, tok_r, tok_d = full
    rng = np.random.default_rng(0)
    n_c = 0
    for j in rng.choice(len(subs), min(80, len(subs)), replace=False):
        s, r = int(subs[j]), int(rels[j])
        got = ids[j][mask[j]]
        ps = past[past[:, 0] == s]
        partners, tri = set(ps[:, 2].tolist()), \
            set(ps[ps[:, 1] == r][:, 2].tolist())
        assert len(set(got.tolist())) == len(got), "duplicate candidate"
        assert set(got.tolist()) <= partners
        if len(tri) <= S2:
            assert tri <= set(got.tolist()), "an (s, r) object is missing"
        if len(partners) <= S1:
            assert partners == set(got.tolist()), "a partner is missing"
        for c, o in enumerate(got):
            e = ps[ps[:, 2] == o]
            e3 = e[e[:, 1] == r]
            st = stat[j][mask[j]][c]
            ref = [np.log1p(len(e3)),
                   np.log1p(t - e3[:, 3].max()) if len(e3) else 0.0,
                   float(len(e3) > 0), np.log1p(len(e)),
                   np.log1p(t - e[:, 3].max())]
            assert np.allclose(st[[0, 1, 3, 4, 5]], ref, atol=1e-5), "stat"
            e = e[np.argsort(e[:, 3], kind="stable")][::-1][:L]
            k = len(e)
            tr, td = tok_r[j][mask[j]][c], tok_d[j][mask[j]][c]
            assert (tr[k:] == R2).all() and (td[:k] == t - e[:, 3]).all(), \
                "stream"
            n_c += 1
    print(f"[ok] brute force: {n_c:,} candidates agree on membership, "
          f"statistics and streams")


if __name__ == "__main__":
    main()
