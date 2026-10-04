#!/usr/bin/env python
"""
Is the TYPED history of an entity pair informative? A data-only test.

No model and no training of any network. Three hand-built scorers rank the
candidates of every test query from events strictly before the query time,
all with the SAME time decay, so the only thing that differs is which events
are allowed to speak and whether their relation type is used:

    recurrence   only past (s, r, o) events            -- the copy family
    blind dyad   every past event between s and o,     -- what a relation-
                 either direction, type ignored           blind pair count sees
    typed dyad   every past event between s and o,
                 weighted by conf(r_e -> r), a relation-to-relation
                 confidence counted on the TRAINING split only

Queries are split into three disjoint strata, defined from the data alone:

    recurrent   the answer already occurred with (s, r)
    dyad_only   it did not, but s and the answer interacted before under
                some other relation or in the other direction
    cold        s and the answer never interacted

If typed beats blind on dyad_only, and beats recurrence on recurrent, then
the relation types of a pair's past events carry ranking information that a
(count, recency) summary throws away. That is the premise ECHO is built on.

    python diagnose_dyad.py --dataset ICEWS18
"""
import argparse
import os
import numpy as np

from echo.data import load_split, augment


def conf_matrix(q, R2, N, window, max_lag=40):
    """
    conf[a, b] = P(a pair that had an a-event sees a b-event within `window`
    time steps afterwards), counted over ordered event pairs on one dyad.
    Row-normalised by how often a occurs, so it is a TLogic-style confidence.
    """
    key = q[:, 0].astype(np.int64) * N + q[:, 2]
    order = np.lexsort((q[:, 3], key))
    key, rel, tim = key[order], q[order, 1], q[order, 3]
    co = np.zeros((R2, R2), np.float64)
    for lag in range(1, max_lag + 1):
        same = key[lag:] == key[:-lag]
        d = tim[lag:] - tim[:-lag]
        ok = same & (d > 0) & (d <= window)
        if not ok.any():
            continue
        np.add.at(co, (rel[:-lag][ok], rel[lag:][ok]), 1.0)
    cnt = np.bincount(rel, minlength=R2).astype(np.float64)
    return co / np.maximum(cnt[:, None], 1.0)


def ranks(score, tgt, filt):
    s = score.astype(np.float64)
    v = s[tgt]
    if len(filt):
        s[filt] = -np.inf
    s[tgt] = v
    better = (s > v).sum()
    tied = (s == v).sum() - 1
    return 1.0 + better + tied / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ICEWS18")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--max_queries", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr, va, te, N, R = load_split(os.path.join(a.data_dir, a.dataset))
    R2 = 2 * R
    times = np.unique(np.concatenate([tr, va, te])[:, 3])
    step = int(np.diff(times).min())
    allq = augment(np.concatenate([tr, va, te]), R)
    trq = augment(tr, R)
    teq = augment(te, R)
    print(f"[{a.dataset}] N={N:,} R={R} step={step}  test queries={len(teq):,}")

    conf = conf_matrix(trq, R2, N, window=7 * step)
    np.fill_diagonal(conf, np.maximum(conf.diagonal(), 1e-6))
    off = conf.copy(); np.fill_diagonal(off, 0)
    print(f"  conf: mean diagonal {conf.diagonal().mean():.4f}   "
          f"mean off-diagonal {off.sum() / (R2 * R2 - R2):.5f}   "
          f"share of mass off the diagonal "
          f"{off.sum() / conf.sum():.3f}")

    # events grouped by subject, time-sorted inside each subject
    order = np.lexsort((allq[:, 3], allq[:, 0]))
    ev = allq[order]
    s_start = np.searchsorted(ev[:, 0], np.arange(N))
    s_end = np.searchsorted(ev[:, 0], np.arange(N), side="right")

    # true answers per (s, r, t) for the time-aware filter
    ans = {}
    for s, r, o, t in allq:
        ans.setdefault((int(s), int(r), int(t)), []).append(int(o))

    rng = np.random.default_rng(a.seed)
    idx = np.arange(len(teq))
    if a.max_queries and a.max_queries < len(idx):
        idx = rng.choice(idx, a.max_queries, replace=False)

    # events grouped by relation, for the subject-free popularity scorer
    order_r = np.lexsort((allq[:, 3], allq[:, 1]))
    evr = allq[order_r]
    r_start = np.searchsorted(evr[:, 1], np.arange(R2))
    r_end = np.searchsorted(evr[:, 1], np.arange(R2), side="right")

    names = ("recurrence", "blind dyad", "typed dyad", "popularity",
             "backoff")
    strata = ("recurrent", "dyad_only", "cold")
    acc = {st: {n: [0.0, 0.0, 0.0] for n in names} for st in strata}
    size = {st: 0 for st in strata}

    for j in idx:
        s, r, o, t = (int(x) for x in teq[j])
        lo, hi = s_start[s], s_end[s]
        blk = ev[lo:hi]
        blk = blk[:np.searchsorted(blk[:, 3], t)]          # strictly before t
        eo, er = blk[:, 2], blk[:, 1]
        d = (t - blk[:, 3]) / step
        w = np.exp(-0.3 * d) + 0.2 * np.exp(-0.02 * d) + 0.02

        on_o = eo == o
        if (on_o & (er == r)).any():
            st = "recurrent"
        elif on_o.any():
            st = "dyad_only"
        else:
            st = "cold"
        size[st] += 1

        filt = np.array([x for x in ans[(s, r, t)] if x != o], np.int64)
        sc = (
            np.bincount(eo, weights=w * (er == r), minlength=N),
            np.bincount(eo, weights=w, minlength=N),
            np.bincount(eo, weights=w * conf[er, r], minlength=N),
        )
        rb = evr[r_start[r]:r_end[r]]
        rb = rb[:np.searchsorted(rb[:, 3], t)]
        dr = (t - rb[:, 3]) / step
        pop = np.bincount(rb[:, 2], minlength=N,
                          weights=np.exp(-0.3 * dr) + 0.2 * np.exp(-0.02 * dr)
                          + 0.02)
        # lexicographic backoff: recurrence, then the pair, then popularity
        nz = lambda x: x / max(x.max(), 1e-12)
        back = nz(sc[0]) + 1e-3 * nz(sc[2]) + 1e-6 * nz(pop)
        sc = sc + (pop, back)
        for n, x in zip(names, sc):
            rk = ranks(x, o, filt)
            m = acc[st][n]
            m[0] += 1.0 / rk; m[1] += rk <= 1; m[2] += rk <= 10

    tot = sum(size.values())
    print(f"\n  {'stratum':<10} {'share':>7} | " + " | ".join(
        f"{n:^20}" for n in names))
    print(f"  {'':<10} {'':>7} | " + " | ".join(
        f"{'MRR':>6} {'H@1':>6} {'H@10':>6}" for _ in names))
    pooled = {n: [0.0, 0.0, 0.0] for n in names}
    for st in strata:
        row = []
        for n in names:
            m = acc[st][n]
            for k in range(3):
                pooled[n][k] += m[k]
            k = max(size[st], 1)
            row.append(f"{100*m[0]/k:>6.2f} {100*m[1]/k:>6.2f} "
                       f"{100*m[2]/k:>6.2f}")
        print(f"  {st:<10} {100*size[st]/tot:>6.1f}% | " + " | ".join(row))
    row = [f"{100*pooled[n][0]/tot:>6.2f} {100*pooled[n][1]/tot:>6.2f} "
           f"{100*pooled[n][2]/tot:>6.2f}" for n in names]
    print(f"  {'ALL':<10} {100.0:>6.1f}% | " + " | ".join(row))
    print("\n  All three scorers use the same decay and see the same events;"
          "\n  conf is counted on the training split only. Ties are averaged.")


if __name__ == "__main__":
    main()
