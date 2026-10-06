"""
ECHO — Event-stream Cross-excitation over dyads with History Ordering.

================================ THE OBSERVATION ============================
Recurrence models ask "did (s, r, o) happen before?". diagnose_dyad.py asks a
wider question of the data alone, with no trained model: what if EVERY past
event between s and o is allowed to speak, under any relation and in either
direction? On ICEWS18 the test queries split into

    recurrent  50.6 %   the answer already occurred with (s, r)
    dyad_only  23.4 %   it did not, but s and the answer interacted before
    cold       26.0 %   the pair never interacted

A copy mechanism is silent on the second stratum by construction. A hand-built
scorer over the pair's events reaches H@10 42 there, and a three-level
backoff (recurrence, pair, popularity) with no trained parameter reaches
MRR 31.8 / H@1 22.6 overall -- the level of DaeMon (31.85 / 22.67).

So the unit that carries the signal is not the triple. It is the DYAD: the
ordered pair (s, o) and the stream of typed, timestamped events on it.

================================== THE MODEL ================================
A query (s, r, ?, t) is a draw from a superposition of four marked point
processes, combined by logaddexp (adding intensities), never by a gate:

    lambda(o) = lambda_struct(o | G_<t, s, r)        every entity
              + lambda_pop   (o | r, t)              every entity
              + lambda_dyad  (o | stream_so, s, r)   the support of s
              + lambda_path  (o | s -> x -> o, r)    two hops from s

lambda_dyad  -- the contribution. For a candidate o the last L events on the
    dyad (s, o) are a sequence of (relation, elapsed time) tokens. The query
    relation is prepended as a read-out token and a small Transformer encodes
    the sequence. One mechanism therefore covers
        self-excitation    past r events on the dyad      (recurrence)
        cross-excitation   past r' events raising r       (temporal rules)
        reciprocity        past events in the other direction
        order effects      r1 then r2 differs from r2 then r1
    and none of them is confined to be monotone in elapsed time, because time
    enters through a sinusoidal encoding of log elapsed time rather than a
    decay. Summary statistics of the full history are concatenated so that
    truncating the stream to L events loses no count.

competition  -- candidates of one query are not scored independently. After
    each candidate is encoded, one attention layer runs ACROSS the support,
    so the intensity of a candidate can depend on what the other candidates'
    streams look like. A query whose answer rotates among a few partners, or
    moves to whichever partner has gone quiet, cannot be expressed by any
    scorer that looks at one candidate at a time. The blocked stratum is
    exactly that case: the answer is in the support and something else in
    the support dominates it.

lambda_path  -- excitation does not stop at the dyad. A dyad-only model is
    silent whenever s and o never interacted (the cold stratum, a quarter to
    a third of ICEWS queries); diagnose_paths.py shows 54 % of those answers
    on ICEWS18 are partners of one of the subject's recent partners, inside a
    set of about 555 entities. Each first-hop candidate x therefore forwards
    its query-conditioned state along its own most recent typed edges
    (x, r2, o), and the messages arriving at o are summed into a path
    intensity. The first-hop state is the dyadic encoding of (s, x), so a
    path is the composition of a full event stream with a typed, timed edge:
    a differentiable form of the length-2 temporal rule, with the rule body
    read from the stream rather than from a single past fact.

lambda_proto -- the path reaches what the subject's partners reach; it does
    not reach an entity that merely RESEMBLES the answers seen before. The
    cold_far stratum (18 % of ICEWS18, MRR 1.6) is exactly that case. The
    support states weight the candidates' entity vectors into one prototype
    of "the kind of object this query has had", and a query vector built
    from the prototype, the subject and the relation is scored against every
    entity by inner product. It is the all-entity counterpart of the global
    history graphs of LogCL and HisRES, with the attention read from the
    streams instead of from a second GNN.

lambda_pop   -- a learned function of multi-scale decayed counts of (r, o)
    and of o, evaluated for ALL entities. It is the relaxed-recurrency signal
    without a top-k cut, and it is what lets a candidate outside the support
    be ranked on evidence rather than on an embedding alone.

lambda_struct -- snapshot evolution (R-GCN with a cross-time link, a GRU over
    snapshots) and a ConvTransE decoder. This is prior work (RE-GCN, DiMNet)
    and is claimed as nothing more than the backbone.

Ablations (train_echo.py): --no_type blanks the relation of every stream
token and keeps its time; --no_stream removes the stream and keeps the
statistics; --compete adds attention across candidates; --no_path,
--no_proto, --no_dyad, --no_pop and --no_struct remove a whole intensity.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from echo.data import N_STAT

NEG = -1e4


# ── structural backbone (prior work) ─────────────────────────────────────────

class SnapLayer(nn.Module):
    """Relational message passing on one snapshot, mean and max aggregated,
    with a link to the same layer depth at the previous snapshot."""

    def __init__(self, d, dropout):
        super().__init__()
        self.w_msg = nn.Linear(d, d, bias=False)
        self.w_self = nn.Linear(d, d, bias=False)
        self.w_cross = nn.Linear(d, d, bias=False)
        self.mix = nn.Linear(2 * d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, R, src, rel, dst, prev):
        msg = self.w_msg(H[src] + R[rel])
        dt, dev = msg.dtype, msg.device
        n, d = H.size(0), msg.size(1)
        mean = torch.zeros(n, d, device=dev, dtype=dt).index_add_(0, dst, msg)
        deg = torch.zeros(n, 1, device=dev, dtype=dt).index_add_(
            0, dst, torch.ones(dst.numel(), 1, device=dev, dtype=dt))
        mean = mean / deg.clamp(min=1.0)
        mx = torch.full((n, d), NEG, device=dev, dtype=dt).index_reduce(
            0, dst, msg, "amax", include_self=True)
        mx = torch.where(deg > 0, mx, torch.zeros_like(mx))
        out = self.mix(torch.cat([mean, mx], -1)) + self.w_self(H)
        if prev is not None:
            out = out + self.w_cross(prev)
        return self.norm(self.drop(F.relu(out)))


class Evolver(nn.Module):
    def __init__(self, d, layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(SnapLayer(d, dropout)
                                    for _ in range(layers))
        self.cell = nn.GRUCell(d, d)
        self.gate = nn.Linear(2 * d, d)

    def forward(self, E0, R, history):
        state, prev = None, [None] * len(self.layers)
        for src, rel, dst in history:
            if src.numel() == 0:
                continue
            H = E0 if state is None else state
            outs = []
            for l, layer in enumerate(self.layers):
                H = layer(H, R, src, rel, dst, prev[l])
                outs.append(H)
            prev = outs
            state = self.cell(H, torch.zeros_like(H) if state is None
                              else state)
        if state is None:
            return E0
        u = torch.sigmoid(self.gate(torch.cat([state, E0.to(state.dtype)], -1)))
        return u * state + (1 - u) * E0


class ConvTransE(nn.Module):
    """LayerNorm instead of BatchNorm: batches are snapshots of very uneven
    size, and they are chunked, so batch statistics would depend on both."""

    def __init__(self, d, channels, dropout):
        super().__init__()
        self.n0 = nn.LayerNorm(d)
        self.conv = nn.Conv1d(2, channels, 3, padding=1)
        self.n1 = nn.LayerNorm(d)
        self.fc = nn.Linear(channels * d, d)
        self.n2 = nn.LayerNorm(d)
        self.d0, self.d1 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, h_s, h_r, table, bias):
        x = self.d0(self.n0(torch.stack([h_s, h_r], 1)))
        x = self.d1(F.relu(self.n1(self.conv(x))))
        x = F.relu(self.n2(self.fc(x.flatten(1))))
        return x @ table.T + bias


# ── dyadic event-stream encoder (the contribution) ──────────────────────────

class TimeEncoding(nn.Module):
    """
    Elapsed time as sinusoids of log(1 + dt) at geometric frequencies, plus
    the raw log. Nothing here is monotone, so the intensity built on it can
    peak at any elapsed time the data asks for.
    """

    def __init__(self, d, n_freq=8):
        super().__init__()
        self.register_buffer(
            "freq", torch.exp(torch.linspace(0.0, math.log(24.0), n_freq)))
        self.proj = nn.Linear(2 * n_freq + 1, d)

    def forward(self, dt):
        x = torch.log1p(dt.float()).unsqueeze(-1)
        a = x * self.freq
        return self.proj(torch.cat([x / 4.0, a.sin(), a.cos()], -1))


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, keep):
        M, T, d = x.shape
        q, k, v = (self.qkv(self.n1(x)).view(M, T, 3, self.h, d // self.h)
                   .permute(2, 0, 3, 1, 4))
        y = F.scaled_dot_product_attention(q, k, v,
                                           attn_mask=keep[:, None, None, :])
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(M, T, d)))
        return x + self.drop(self.ff(self.n2(x)))


class StreamEncoder(nn.Module):
    """[query relation] + L (relation, elapsed time) tokens -> one vector."""

    def __init__(self, R2, d_ent, cfg):
        super().__init__()
        d = cfg.stream_dim
        self.R2 = R2
        self.no_type = cfg.no_type
        self.tok = nn.Embedding(R2 + 1, d, padding_idx=R2)
        self.time = TimeEncoding(d)
        self.query = nn.Embedding(R2, d)
        self.sub = nn.Linear(d_ent, d)
        self.blocks = nn.ModuleList(
            Block(d, cfg.stream_heads, cfg.dropout)
            for _ in range(cfg.stream_layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, rels, h_sub, tok_r, tok_d):
        """rels (M,), h_sub (M, d_ent), tok_r / tok_d (M, L)."""
        tok_r = tok_r.long()
        keep = tok_r != self.R2
        if self.no_type:
            tok_r = torch.where(keep, torch.zeros_like(tok_r), tok_r)
        x = self.tok(tok_r) + self.time(tok_d) * keep.unsqueeze(-1)
        q = self.query(rels) + self.sub(h_sub)
        x = torch.cat([q.unsqueeze(1), x.to(q.dtype)], 1)
        keep = torch.cat([torch.ones_like(keep[:, :1]), keep], 1)
        for b in self.blocks:
            x = b(x, keep)
        return self.norm(x[:, 0])


# ── ECHO ─────────────────────────────────────────────────────────────────────

class Echo(nn.Module):
    TAUS = (1.0, 4.0, 16.0, 64.0)          # decay scales of the popularity field

    def __init__(self, num_entities, num_relations, cfg, ev, t_ptr):
        super().__init__()
        d, ds = cfg.embed_dim, cfg.stream_dim
        self.N, self.R2 = num_entities, 2 * num_relations
        self.cfg = cfg
        self.H = cfg.hist_len

        # the time-sorted event table lives on the model's device; snapshot
        # graphs and the popularity field are slices of it
        ev = torch.as_tensor(ev)
        self.register_buffer("ev_s", ev[:, 0].contiguous(), persistent=False)
        self.register_buffer("ev_r", ev[:, 1].contiguous(), persistent=False)
        self.register_buffer("ev_o", ev[:, 2].contiguous(), persistent=False)
        self.register_buffer("ev_t", ev[:, 3].contiguous(), persistent=False)
        self.t_ptr = [int(x) for x in t_ptr]
        self.register_buffer("taus", torch.tensor(self.TAUS),
                             persistent=False)

        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(self.R2, d)
        self.ent_bias = nn.Parameter(torch.zeros(num_entities))
        nn.init.xavier_normal_(self.ent_emb.weight)
        nn.init.xavier_normal_(self.rel_emb.weight)

        self.evolver = Evolver(d, cfg.gcn_layers, cfg.dropout)
        self.decoder = ConvTransE(d, cfg.conv_channels, cfg.dropout)

        # popularity field
        K = 2 * (len(self.TAUS) + 1)
        self.pop_in = nn.Linear(K, cfg.pop_dim)
        self.pop_rel = nn.Linear(d, cfg.pop_dim)
        self.pop_out = nn.Linear(cfg.pop_dim, 1)
        self.pop_bias = nn.Parameter(torch.tensor(cfg.bias_init))

        # dyadic intensity
        self.stream = StreamEncoder(self.R2, d, cfg)
        self.stat_norm = nn.LayerNorm(N_STAT)
        self.c_ent = nn.Linear(d, ds)
        self.c_sub = nn.Linear(d, ds)
        self.c_rel = nn.Linear(d, ds)
        self.trunk = nn.Sequential(
            nn.Linear(4 * ds + N_STAT, 2 * ds), nn.LayerNorm(2 * ds),
            nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(2 * ds, 2 * ds), nn.LayerNorm(2 * ds), nn.GELU())
        self.dyad_out = nn.Linear(2 * ds, 1)
        nn.init.normal_(self.dyad_out.weight, std=0.02)
        nn.init.zeros_(self.dyad_out.bias)
        self.dyad_bias = nn.Parameter(torch.tensor(cfg.bias_init))

        # competition across the candidates of one query
        self.compete = Block(
            2 * ds, cfg.stream_heads, cfg.dropout) if cfg.compete else None

        # two-hop path intensity
        k = cfg.path_dim
        self.p_src = nn.Linear(2 * ds, k)
        self.p_rel = nn.Embedding(self.R2, k)
        self.p_time = TimeEncoding(k)
        self.p_cnt = nn.Linear(1, k)
        self.p_edge = nn.Sequential(nn.LayerNorm(k), nn.GELU(),
                                    nn.Linear(k, k))
        self.p_norm = nn.LayerNorm(k)
        self.p_trunk = nn.Sequential(
            nn.Linear(k + 1 + 3 * ds, 2 * ds), nn.LayerNorm(2 * ds),
            nn.GELU(), nn.Dropout(cfg.dropout))
        self.p_out = nn.Linear(2 * ds, 1)
        nn.init.normal_(self.p_out.weight, std=0.02)
        nn.init.zeros_(self.p_out.bias)
        self.path_bias = nn.Parameter(torch.tensor(cfg.bias_init))

        # prototype intensity
        self.proto_att = nn.Linear(2 * ds, 1)
        self.proto_q = nn.Sequential(
            nn.Linear(3 * d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(d, d))
        self.proto_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(d)))
        self.proto_bias = nn.Parameter(torch.tensor(cfg.bias_init))

    # ── per-timestamp context: computed once, shared by every query chunk ───

    def history(self, t):
        out = []
        for tt in range(max(t - self.H, 0), t):
            a, b = self.t_ptr[tt], self.t_ptr[tt + 1]
            out.append((self.ev_s[a:b], self.ev_r[a:b], self.ev_o[a:b]))
        return out

    def evolve(self, t):
        if self.cfg.no_struct:
            return self.ent_emb.weight
        return self.evolver(self.ent_emb.weight, self.rel_emb.weight,
                            self.history(t))

    @torch.no_grad()
    def pop_features(self, t, rel_u):
        """
        log1p of decayed counts before t, at every scale in TAUS and
        undecayed: (Ru, N, K/2) for (r, o) and (N, K/2) for o alone.
        """
        e = self.t_ptr[t]
        er, eo = self.ev_r[:e], self.ev_o[:e]
        dt = (t - self.ev_t[:e]).float().unsqueeze(1)
        w = torch.cat([torch.exp(-(dt - 1.0) / self.taus),
                       torch.ones_like(dt)], 1)
        k = w.size(1)
        act = torch.zeros(self.N, k, device=w.device).index_add_(0, eo, w)
        rmap = torch.full((self.R2,), -1, dtype=torch.long, device=w.device)
        rmap[rel_u] = torch.arange(rel_u.numel(), device=w.device)
        row = rmap[er]
        sel = row >= 0
        ro = torch.zeros(rel_u.numel() * self.N, k, device=w.device)
        ro.index_add_(0, row[sel] * self.N + eo[sel], w[sel])
        return torch.log1p(ro.view(rel_u.numel(), self.N, k)), torch.log1p(act)

    def popularity(self, t, rel_u):
        """log lambda_pop for every (relation in rel_u, entity): (Ru, N)."""
        ro, act = self.pop_features(t, rel_u)
        x = torch.cat([ro, act.unsqueeze(0).expand(ro.size(0), -1, -1)], -1)
        h = self.pop_in(x) + self.pop_rel(self.rel_emb(rel_u)).unsqueeze(1)
        return self.pop_out(F.gelu(h)).squeeze(-1) + self.pop_bias

    # ── forward over one chunk of queries ────────────────────────────────────

    def forward(self, E, pop, subs, rels, sup_ids, sup_stat, sup_mask,
                tok_r, tok_d, sup_v=None, edges=None, return_struct=False):
        """
        E      (N, d)   evolved entity states for this timestamp
        pop    (B, N)   log lambda_pop rows for these queries, or None
        sup_v  (B, S)   row of each candidate in the edge table
        edges  dict     e_dst / e_rel / e_dt / e_cnt / e_mask, (V, P2) each
        Returns log-intensities over all N entities, (B, N) float32.
        """
        cfg = self.cfg
        B = subs.numel()
        if cfg.no_struct:
            f_struct = self.ent_bias.float().unsqueeze(0).expand(B, -1)
        else:
            f_struct = self.decoder(E[subs], self.rel_emb(rels), E,
                                    self.ent_bias).float()
        out = f_struct if pop is None else torch.logaddexp(f_struct,
                                                           pop.float())
        rows, slots = sup_mask.nonzero(as_tuple=True)
        if cfg.no_dyad or rows.numel() == 0:
            return (out, f_struct) if return_struct else out

        # ── first hop: one state per (query, candidate) ─────────────────────
        ids = sup_ids[rows, slots]
        h_s = E[subs]
        if cfg.no_stream:
            z = E.new_zeros(rows.numel(), cfg.stream_dim)
        else:
            z = self.stream(rels[rows], h_s[rows],
                            tok_r[rows, slots], tok_d[rows, slots])
        q_sub = self.c_sub(h_s).to(z.dtype)
        q_rel = self.c_rel(self.rel_emb(rels)).to(z.dtype)
        h = self.trunk(torch.cat([
            z, self.stat_norm(sup_stat[rows, slots]).to(z.dtype),
            self.c_ent(E[ids]).to(z.dtype), q_sub[rows], q_rel[rows]], -1))

        # ── competition across the support of each query ────────────────────
        if self.compete is not None:
            pad = h.new_zeros(B, sup_mask.size(1), h.size(-1))
            pad[rows, slots] = h
            # a query with an empty support must still attend to something
            keep = sup_mask | ~sup_mask.any(1, keepdim=True)
            h = self.compete(pad, keep)[rows, slots].to(h.dtype)

        f = self.dyad_out(h).squeeze(-1) + self.dyad_bias
        out = out.index_put((rows, ids),
                            torch.logaddexp(out[rows, ids], f.float()))

        # ── prototype: "like the ones before", scored over ALL entities ─────
        if not cfg.no_proto:
            att = self.proto_att(h).squeeze(-1).float()
            rmax = att.new_full((B,), float("-inf")).scatter_reduce(
                0, rows, att, "amax")
            w = torch.exp(att - rmax[rows])        # softmax within each row
            den = w.new_zeros(B).index_add_(0, rows, w)
            w = w / den[rows].clamp(min=1e-12)
            proto = E.new_zeros(B, E.size(1)).index_add_(
                0, rows, (E[ids].float() * w.unsqueeze(1)).to(E.dtype))
            has = (den > 0).unsqueeze(1).to(E.dtype)
            qp = self.proto_q(torch.cat(
                [proto * has, h_s, self.rel_emb(rels).to(E.dtype)], -1))
            fq = (qp @ E.T).float() * self.proto_scale + self.proto_bias
            out = torch.logaddexp(out, fq)

        # ── second hop: forward each state along the candidate's own edges ──
        if not cfg.no_path and edges is not None:
            vi = sup_v[rows, slots]
            pr, pj = edges["e_mask"][vi].nonzero(as_tuple=True)
            v = vi[pr]
            dst, qb = edges["e_dst"][v, pj], rows[pr]
            ok = dst != subs[qb]                   # not back to the subject
            pr, pj, v, dst, qb = pr[ok], pj[ok], v[ok], dst[ok], qb[ok]
            if pr.numel():
                edge = self.p_edge(
                    self.p_rel(edges["e_rel"][v, pj].long())
                    + self.p_time(edges["e_dt"][v, pj])
                    + self.p_cnt(edges["e_cnt"][v, pj].unsqueeze(-1)))
                msg = self.p_src(h)[pr] * edge.to(h.dtype)
                key, inv = torch.unique(qb * self.N + dst,
                                        return_inverse=True)
                m = msg.new_zeros(key.numel(), msg.size(1)).index_add_(
                    0, inv, msg)
                n = msg.new_zeros(key.numel()).index_add_(
                    0, inv, torch.ones_like(msg[:, 0]))
                ub, uo = key // self.N, key % self.N
                x = torch.cat([
                    self.p_norm(m / n.unsqueeze(1).sqrt()).to(h.dtype),
                    torch.log1p(n).unsqueeze(1).to(h.dtype),
                    self.c_ent(E[uo]).to(h.dtype), q_sub[ub], q_rel[ub]], -1)
                fp = self.p_out(self.p_trunk(x)).squeeze(-1) + self.path_bias
                out = out.index_put((ub, uo),
                                    torch.logaddexp(out[ub, uo], fp.float()))
        return (out, f_struct) if return_struct else out
