#!/usr/bin/env python
"""
Can the COLD stratum be reached in two hops? A data-only test.

A query (s, r, ?, t) is cold when s and the answer never interacted before t.
Nothing on the dyad (s, answer) exists, so the dyadic intensity is silent.
This script asks whether the answer is nevertheless a partner of one of the
subject's partners,

        s  --  x  --  o        x in the P1 most recent partners of s
                               o in the P2 most recent partners of x

and how well a hand-built path counter ranks it. No network is trained.

    path score(o) = sum over x of  w(s, x) * w(x, o)
    w(a, b)       = sum over past events on the dyad (a, b) of the same decay
                    used in diagnose_dyad.py

    python diagnose_paths.py --dataset ICEWS18
"""
import argparse
import os
import numpy as np

from echo.data import load_split, augment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ICEWS18")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--max_queries", type=int, default=20000)
    ap.add_argument("--p1", type=int, default=96)
    ap.add_argument("--p2", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr, va, te, N, R = load_split(os.path.join(a.data_dir, a.dataset))
    R2 = 2 * R
    times = np.unique(np.concatenate([tr, va, te])[:, 3])
    step = int(np.diff(times).min())
    allq = augment(np.concatenate([tr, va, te]), R)
    teq = augment(te, R)

    order = np.lexsort((allq[:, 3], allq[:, 0]))
    ev = allq[order]
    s_start = np.searchsorted(ev[:, 0], np.arange(N))
    s_end = np.searchsorted(ev[:, 0], np.arange(N), side="right")
    order_r = np.lexsort((allq[:, 3], allq[:, 1]))
    evr = allq[order_r]
    r_start = np.searchsorted(evr[:, 1], np.arange(R2))
    r_end = np.searchsorted(evr[:, 1], np.arange(R2), side="right")
    ans = {}
    for s, r, o, t in allq:
        ans.setdefault((int(s), int(r), int(t)), []).append(int(o))

    def decay(d):
        return np.exp(-0.3 * d) + 0.2 * np.exp(-0.02 * d) + 0.02

    cache = {}

    def partners(x, t, k):
        """Top-k partners of x before t by recency, with dyad weights."""
        key = (x, t, k)
        if key not in cache:
            blk = ev[s_start[x]:s_end[x]]
            blk = blk[:np.searchsorted(blk[:, 3], t)]
            if len(blk) == 0:
                cache[key] = (np.zeros(0, np.int64), np.zeros(0))
            else:
                w = np.bincount(blk[:, 2], minlength=N,
                                weights=decay((t - blk[:, 3]) / step))
                last = np.full(N, -1.0)
                last[blk[:, 2]] = blk[:, 3]           # time-sorted: last wins
                ids = np.flatnonzero(last >= 0)
                if len(ids) > k:
                    ids = ids[np.argpartition(-last[ids], k - 1)[:k]]
                cache[key] = (ids, w[ids])
        return cache[key]

    def rank(score, tgt, filt):
        s = score.astype(np.float64)
        v = s[tgt]
        if len(filt):
            s[filt] = -np.inf
        s[tgt] = v
        return 1.0 + (s > v).sum() + ((s == v).sum() - 1) / 2.0

    rng = np.random.default_rng(a.seed)
    idx = np.arange(len(teq))
    if a.max_queries and a.max_queries < len(idx):
        idx = rng.choice(idx, a.max_queries, replace=False)

    names = ("popularity", "paths", "paths x popularity")
    acc = {n: np.zeros(3) for n in names}
    n_cold = n_reach = n_all = 0
    cand = []
    for j in idx:
        s, r, o, t = (int(x) for x in teq[j])
        n_all += 1
        blk = ev[s_start[s]:s_end[s]]
        blk = blk[:np.searchsorted(blk[:, 3], t)]
        if (blk[:, 2] == o).any():
            continue                                   # not cold
        n_cold += 1
        xs, w1 = partners(s, t, a.p1)
        path = np.zeros(N)
        for x, wx in zip(xs, w1):
            os_, w2 = partners(int(x), t, a.p2)
            path[os_] += wx * w2
        path[s] = 0.0
        n_reach += path[o] > 0
        cand.append((path > 0).sum())

        rb = evr[r_start[r]:r_end[r]]
        rb = rb[:np.searchsorted(rb[:, 3], t)]
        pop = np.bincount(rb[:, 2], minlength=N,
                          weights=decay((t - rb[:, 3]) / step))
        filt = np.array([x for x in ans[(s, r, t)] if x != o], np.int64)
        both = np.log1p(path / max(path.max(), 1e-12) * 50) * \
            np.log1p(pop / max(pop.max(), 1e-12) * 50) + 1e-6 * pop
        for n, sc in zip(names, (pop, path, both)):
            rk = rank(sc, o, filt)
            acc[n] += (1.0 / rk, rk <= 1, rk <= 10)

    print(f"[{a.dataset}] {n_all:,} test queries sampled, P1={a.p1} "
          f"P2={a.p2}")
    print(f"  cold queries            {100*n_cold/n_all:5.1f} % of all")
    print(f"  answer within two hops  {100*n_reach/max(n_cold,1):5.1f} % "
          f"of cold   (mean {np.mean(cand):,.0f} entities reached, "
          f"{100*np.mean(cand)/N:.1f} % of all entities)")
    print(f"\n  on the cold stratum       MRR    H@1   H@10")
    for n in names:
        m = 100 * acc[n] / max(n_cold, 1)
        print(f"  {n:<22} {m[0]:>6.2f} {m[1]:>6.2f} {m[2]:>6.2f}")


if __name__ == "__main__":
    main()
