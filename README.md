# ECHO — dyadic event-stream intensity for temporal knowledge graph forecasting

ECHO predicts `(s, r, ?, t)` from everything that happened before `t`. Its
unit of evidence is the **dyad**: the ordered pair `(s, o)` and the stream of
typed, timestamped events on it, under any relation and in either direction.

## Why the dyad

`diagnose_dyad.py` ranks the candidates of every test query with hand-built
counters. No network is trained. All scorers share one time decay and read
only events before the query time.

| scorer | what may speak |
|---|---|
| recurrence | past `(s, r, o)` events only, which is the copy family |
| typed dyad | every past event between `s` and `o`, weighted by a relation-to-relation confidence counted on the training split |
| backoff | recurrence, then the dyad, then the popularity of `o` under `r` |

Time-aware filtered, ties averaged, x100:

| dataset | scorer | MRR | H@1 | H@10 |
|---|---|---|---|---|
| ICEWS14s | recurrence | 37.08 | 30.80 | 48.50 |
| ICEWS14s | typed dyad | 42.72 | 33.73 | 60.39 |
| ICEWS14s | backoff | 43.26 | 34.08 | 60.85 |
| ICEWS18 (30k queries) | recurrence | 28.02 | 20.93 | 41.41 |
| ICEWS18 (30k queries) | typed dyad | 30.65 | 21.05 | 49.82 |
| ICEWS18 (30k queries) | backoff | 31.82 | 22.55 | 49.88 |

For scale, published trained models on ICEWS18 report DaeMon 31.85 / 22.67 /
49.80 and RE-GCN 30.58 / 21.01 / 48.75. A counter with no trained parameter
is at that level once it is allowed to read the dyad.

The test queries split into three strata, defined from the data alone:

| dataset | recurrent | dyad_only | cold |
|---|---|---|---|
| ICEWS14s | 52.4 % | 22.0 % | 25.7 % |
| ICEWS18 | 50.6 % | 23.4 % | 26.0 % |
| WIKI | 86.9 % | 0.1 % | 13.0 % |
| YAGO | 92.7 % | 0.1 % | 7.3 % |

`dyad_only` means the answer never occurred with `(s, r)`, but `s` and the
answer interacted before. A copy mechanism is silent there by construction.
It is a quarter of the ICEWS queries, and the dyad counters reach H@10 42
(ICEWS18) and 60 (ICEWS14s) on it.

Two things this measurement does **not** show, stated so nobody over-reads it:

- On `dyad_only` the typed counter is no better than a type-blind one
  (ICEWS18 MRR 20.33 against 20.01). The gain of the dyad comes from reading
  the pair at all. Whether a learned model extracts more from the types is
  what the `--no_type` ablation tests.
- On YAGO and WIKI the `dyad_only` stratum is empty. ECHO can only help
  there through the order and timing of the stream, not through new coverage.

## The model

Three intensities over the candidate entities, added as intensities
(`logaddexp`), with no gate between them:

```
lambda(o) = lambda_struct(o | G_<t, s, r)      every entity
          + lambda_pop(o | r, t)               every entity
          + lambda_dyad(o | stream_so, s, r)   the support of s
```

- **`lambda_dyad` is the contribution.** The last `L` events on the dyad are
  tokens `(relation, elapsed time)`. The query relation is prepended as a
  read-out token and a two-layer Transformer encodes the sequence. One
  mechanism covers recurrence, cross-relation excitation (temporal rules),
  reciprocity and order effects. Time enters as sinusoids of log elapsed
  time, so the intensity is not confined to decay monotonically.
- **`lambda_pop`** is a learned function of multi-scale decayed counts of
  `(r, o)` and of `o`, for all entities, with no top-k cut.
- **`lambda_struct`** is snapshot evolution with a ConvTransE decoder. It is
  prior work (RE-GCN, DiMNet) and is only the backbone.

Relation to prior work: the type-to-type counter above is close to the
length-1 rules of TLogic and to the rule confidences of CountTRuCoLa. ECHO
replaces independent rule confidences with a sequence model over the dyad's
events and superposes it with a structural encoder.

## Run

```bash
./get_data.sh
python verify.py --dataset ICEWS14s
python diagnose_dyad.py --dataset ICEWS18
python train_echo.py --dataset ICEWS18 --gpu 0
```

Three seeds per dataset, then the ablations, then the table:

```bash
GPUS="0 1 2 3" ./run_all.sh
GPUS="0 1 2 3" ABLATE=ICEWS18 ./run_all.sh
python collect.py
```

Training runs on one GPU per process. `--gpu N` picks the card and
`--gpu -1` takes the one with the most free memory. `--device cuda` refuses
to start without CUDA instead of falling back to CPU. bf16 autocast is on by
default on the GPU; `--no_amp` turns it off.

Ablations: `--no_type` (stream keeps times, loses relation types),
`--no_stream` (statistics only), `--no_dyad`, `--no_pop`, `--no_struct`.

`verify.py` checks that deleting every event at or after `t` changes nothing
the model reads at `t`, and compares the vectorised support builder with a
brute-force loop.

## Protocol

Validation MRR selects the checkpoint. Test reports raw and time-aware
filtered ranks. Ties resolve to their average rank. Time is the snapshot
index, so one step is one snapshot on every dataset. The history before a
test timestamp is the true history, which is the RE-GCN setting.

## Status

No trained result is claimed yet. What has been checked so far:

- `verify.py` passes on ICEWS14s: no leakage, and the support builder agrees
  with brute force on 4,645 candidates.
- The full pipeline trains and evaluates end to end on CPU.
- The diagnostic numbers above are measured.

GPU runs with three seeds per dataset are the next step. A margin over the
published numbers is only a result once it holds across seeds.

Data is not tracked in git. Put each dataset in `data/<NAME>/` with
`train.txt`, `valid.txt` and `test.txt`, or run `./get_data.sh`.
