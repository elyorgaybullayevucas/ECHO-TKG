"""
Data pipeline for ECHO.

Time is re-indexed to 0..T-1 (the rank of the timestamp on the full
timeline), so "one step" always means "one snapshot" whatever the raw unit
is (24 on ICEWS, 15 on GDELT, 1 on YAGO / WIKI). Every elapsed time below is
a number of snapshots.

One training item is one TIMESTAMP: all queries at t, and for each query a
support of candidate objects taken from the subject's own past:

    triple part   objects that already occurred with (s, r)
    dyad part     the subject's most recent partners under ANY relation and
                  in EITHER direction

and for every candidate

    stat    eight summary statistics of the (s, r, o) and (s, ., o) histories
    stream  the last L events on the dyad (s, o), each a (relation, elapsed
            time) pair, most recent first

The stream is what is new here. It is independent of the query relation, it
keeps the TYPE of every past event, and it is the input of the dyadic
intensity in echo/model.py.

Everything is computed with vectorised numpy over all subjects of a
timestamp at once; there is no Python loop over queries or candidates, and
nothing later than t-1 is ever read.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

N_STAT = 8


def _read(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                rows.append((int(p[0]), int(p[1]), int(p[2]), int(p[3])))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 4)


def load_split(base):
    tr, va, te = (_read(os.path.join(base, f"{s}.txt"))
                  for s in ("train", "valid", "test"))
    allq = np.concatenate([tr, va, te], 0)
    N = int(allq[:, [0, 2]].max()) + 1
    R = int(allq[:, 1].max()) + 1
    return tr, va, te, N, R


def augment(q, R):
    inv = np.stack([q[:, 2], q[:, 1] + R, q[:, 0], q[:, 3]], 1)
    return np.concatenate([q, inv], 0)


class EchoIndex:
    """Flat, fork-safe arrays; no Python dict on the query path."""

    def __init__(self, aug, N, R2, T):
        self.N, self.R2, self.T = N, R2, T
        o = np.lexsort((aug[:, 3], aug[:, 0]))            # by (s, t)
        self.es, self.er = aug[o, 0], aug[o, 1]
        self.eo, self.et = aug[o, 2], aug[o, 3]
        self.gk = self.es * T + self.et

        k = (aug[:, 0] * R2 + aug[:, 1]) * T + aug[:, 3]  # (s, r, t)
        o2 = np.argsort(k, kind="stable")
        self.a_key, self.a_obj = k[o2], aug[o2, 2]

    def answers(self, subs, rels, t):
        """All true objects of (s, r, t): (row index, object) pairs."""
        k = (subs * self.R2 + rels) * self.T + t
        lo = np.searchsorted(self.a_key, k)
        hi = np.searchsorted(self.a_key, k, side="right")
        n = hi - lo
        tot = int(n.sum())
        off = np.cumsum(n) - n
        pos = np.repeat(lo - off, n) + np.arange(tot)
        return np.repeat(np.arange(len(k)), n), self.a_obj[pos]


def _groups(sorted_key):
    first = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    return first, np.r_[first[1:], len(sorted_key)]


def build_support(ix, t, subs, rels, S1, S2, L, horizon=0):
    """
    Supports, statistics and dyad streams for all queries (subs, rels) at
    time index t. Returns numpy arrays

        ids   (B, S) int64      stat (B, S, N_STAT) float32
        mask  (B, S) bool       tok_r (B, S, L) int16   (R2 = padding)
                                tok_d (B, S, L) int32   elapsed snapshots
    """
    N, R2, T = ix.N, ix.R2, ix.T
    B = len(subs)
    U, inv = np.unique(subs, return_inverse=True)
    t0 = max(t - horizon, 0) if horizon > 0 else 0
    lo = np.searchsorted(ix.gk, U * T + t0)
    hi = np.searchsorted(ix.gk, U * T + t)                # strictly before t
    lens = hi - lo
    tot = int(lens.sum())
    if tot == 0:
        return (np.zeros((B, 1), np.int64),
                np.zeros((B, 1, N_STAT), np.float32),
                np.zeros((B, 1), bool),
                np.full((B, 1, L), R2, np.int16),
                np.zeros((B, 1, L), np.int32))

    off = np.cumsum(lens) - lens
    rows = np.repeat(lo - off, lens) + np.arange(tot)
    su = np.repeat(np.arange(len(U)), lens)
    eo, er, et = ix.eo[rows], ix.er[rows], ix.et[rows]

    # ── dyads (subject, object): events stay time-ordered inside a dyad ─────
    dk = su * N + eo
    od = np.argsort(dk, kind="stable")
    dk_s, dt_s, dr_s = dk[od], et[od], er[od]
    d_first, d_end = _groups(dk_s)
    d_key = dk_s[d_first]
    d_cnt = d_end - d_first
    d_last, d_t0 = dt_s[d_end - 1], dt_s[d_first]
    d_sub, d_obj = d_key // N, d_key % N
    o2 = np.lexsort((-d_last, d_sub))                     # most recent first
    sub_first = np.searchsorted(d_sub[o2], np.arange(len(U)))
    sub_n = np.bincount(d_sub, minlength=len(U))

    # ── triples (subject, relation, object) ─────────────────────────────────
    tk = (su * R2 + er) * N + eo
    ot = np.argsort(tk, kind="stable")
    tk_s, tt_s = tk[ot], et[ot]
    t_first, t_end = _groups(tk_s)
    t_key = tk_s[t_first]
    t_cnt = t_end - t_first
    t_last, t_t0 = tt_s[t_end - 1], tt_s[t_first]
    t_grp = t_key // N
    o3 = np.lexsort((-t_last, t_grp))
    grp_sorted = t_grp[o3]
    t_rank = np.empty(len(t_key), np.int64)
    t_rank[o3] = np.arange(len(o3)) - np.searchsorted(grp_sorted, grp_sorted)

    # ── triple part: the S2 most recent objects of (s, r) ───────────────────
    g = inv * R2 + rels
    qa = np.searchsorted(grp_sorted, g)
    qb = np.searchsorted(grp_sorted, g, side="right")
    j2 = np.arange(S2)[None, :]
    m2 = j2 < np.minimum(qb - qa, S2)[:, None]
    tri2 = o3[np.where(m2, qa[:, None] + j2, 0)]
    cand2 = t_key[tri2] % N
    dy2 = np.searchsorted(d_key, inv[:, None] * N + cand2)
    dy2 = np.minimum(dy2, len(d_key) - 1)

    # ── dyad part: the S1 most recent partners of s ─────────────────────────
    j1 = np.arange(S1)[None, :]
    m1 = j1 < np.minimum(sub_n[inv], S1)[:, None]
    dy1 = o2[np.where(m1, sub_first[inv][:, None] + j1, 0)]
    cand1 = d_obj[dy1]
    key1 = g[:, None] * N + cand1
    tri1 = np.minimum(np.searchsorted(t_key, key1), len(t_key) - 1)
    hit1 = (t_key[tri1] == key1) & m1
    m1 &= ~(hit1 & (t_rank[tri1] < S2))                   # already in part 2

    ids = np.concatenate([cand2, cand1], 1)
    mask = np.concatenate([m2, m1], 1)
    dy = np.concatenate([dy2, dy1], 1)
    tri = np.concatenate([tri2, tri1], 1)
    has = np.concatenate([m2, hit1], 1) & mask

    # left-pack and cut to the widest row
    order = np.argsort(~mask, axis=1, kind="stable")
    W = max(1, int(mask.sum(1).max()))
    order = order[:, :W]
    ids, mask, dy, tri, has = (np.take_along_axis(x, order, 1)
                               for x in (ids, mask, dy, tri, has))
    ids = np.where(mask, ids, 0)

    # ── statistics ──────────────────────────────────────────────────────────
    c3 = np.where(has, t_cnt[tri], 0).astype(np.float32)
    e3 = np.where(has, t - t_last[tri], 0).astype(np.float32)
    g3 = np.where(c3 > 1, (t_last[tri] - t_t0[tri]) / np.maximum(c3 - 1, 1), 0)
    c2 = d_cnt[dy].astype(np.float32)
    e2 = (t - d_last[dy]).astype(np.float32)
    g2 = np.where(c2 > 1, (d_last[dy] - d_t0[dy]) / np.maximum(c2 - 1, 1), 0)
    stat = np.stack([np.log1p(c3), np.log1p(e3), np.log1p(g3),
                     has.astype(np.float32),
                     np.log1p(c2), np.log1p(e2), np.log1p(g2),
                     c3 / np.maximum(c2, 1)], -1).astype(np.float32)
    stat *= mask[..., None]

    # ── streams: last L events of the dyad, most recent first ───────────────
    pos = d_end[dy][..., None] - 1 - np.arange(L)
    ok = (pos >= d_first[dy][..., None]) & mask[..., None]
    pos = np.where(ok, pos, 0)
    tok_r = np.where(ok, dr_s[pos], R2).astype(np.int16)
    tok_d = np.where(ok, t - dt_s[pos], 0).astype(np.int32)
    return ids, stat, mask, tok_r, tok_d


class SnapshotSet(Dataset):
    """One item = one timestamp: its queries (both directions) and supports."""

    def __init__(self, quads, ix, R, cfg):
        self.ix, self.cfg = ix, cfg
        q = augment(quads, R)
        q = q[np.argsort(q[:, 3], kind="stable")]
        self.q = q
        self.times = np.unique(q[:, 3])
        self.starts = np.searchsorted(q[:, 3], self.times)
        self.ends = np.append(self.starts[1:], len(q))

    def __len__(self):
        return len(self.times)

    def __getitem__(self, i):
        c = self.cfg
        t = int(self.times[i])
        blk = self.q[self.starts[i]:self.ends[i]]
        subs, rels, objs = blk[:, 0], blk[:, 1], blk[:, 2]
        ids, stat, mask, tok_r, tok_d = build_support(
            self.ix, t, subs, rels, c.dyad_support, c.triple_support,
            c.stream_len, c.horizon)
        f = torch.from_numpy
        return dict(t=t, subs=f(subs), rels=f(rels), objs=f(objs),
                    sup_ids=f(ids), sup_stat=f(stat), sup_mask=f(mask),
                    tok_r=f(tok_r), tok_d=f(tok_d))


def identity_collate(batch):
    return batch[0]


class EchoData:
    def __init__(self, cfg):
        base = os.path.join(cfg.data_dir, cfg.dataset)
        tr, va, te, N, R = load_split(base)
        allq = np.concatenate([tr, va, te], 0)
        timeline = np.unique(allq[:, 3])

        def tidx(q):
            q = q.copy()
            q[:, 3] = np.searchsorted(timeline, q[:, 3])
            return q

        tr, va, te = tidx(tr), tidx(va), tidx(te)
        self.num_entities, self.num_relations = N, R
        self.T = len(timeline)
        aug = augment(np.concatenate([tr, va, te], 0), R)
        self.index = EchoIndex(aug, N, 2 * R, self.T)

        # time-sorted event table; the model keeps a copy on its device and
        # slices it for the snapshot graphs and the popularity field
        o = np.argsort(aug[:, 3], kind="stable")
        self.ev = aug[o]
        self.t_ptr = np.searchsorted(self.ev[:, 3], np.arange(self.T + 1))

        self.train_set = SnapshotSet(tr, self.index, R, cfg)
        self.valid_set = SnapshotSet(va, self.index, R, cfg)
        self.test_set = SnapshotSet(te, self.index, R, cfg)
        print(f"[{cfg.dataset}] entities={N:,} relations={R} "
              f"snapshots={self.T:,}  train={len(tr):,} valid={len(va):,} "
              f"test={len(te):,}  "
              f"avg {len(aug)/self.T:,.0f} edges/snapshot")
