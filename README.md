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

Four intensities over the candidate entities, added as intensities
(`logaddexp`), with no gate between them:

```
lambda(o) = lambda_struct(o | G_<t, s, r)      every entity
          + lambda_pop(o | r, t)               every entity
          + lambda_dyad(o | stream_so, s, r)   the support of s
          + lambda_path(o | s -> x -> o, r)    two hops from s
```

- **`lambda_dyad`.** The last `L` events on the dyad are tokens
  `(relation, elapsed time)`. The query relation is prepended as a read-out
  token and a two-layer Transformer encodes the sequence. One mechanism
  covers recurrence, cross-relation excitation (temporal rules), reciprocity
  and order effects. Time enters as sinusoids of log elapsed time, so the
  intensity is not confined to decay monotonically.
- **Competition.** One attention layer runs across the candidates of a
  query, so a candidate's intensity can depend on the other candidates'
  streams. A scorer that looks at one candidate at a time cannot express
  "the answer rotates among these partners".
- **`lambda_path`.** Each first-hop candidate `x` forwards its
  query-conditioned state along its own most recent typed edges `(x, r2, o)`.
  Messages arriving at `o` are summed into a path intensity. A path is the
  composition of a full event stream with a typed, timed edge.
- **`lambda_proto`.** The support states weight the candidates' entity
  vectors into a prototype of "the kind of object this query has had"; a
  query vector from the prototype, the subject and the relation is scored
  against every entity. It reaches entities that resemble past answers
  without being connected to them, which the path cannot.
- **Candidate context.** A candidate's own most recent typed edges, the
  table the path already uses, are pooled into one vector and fed to the
  dyadic trunk: what the candidate has been doing with anyone.
- **`lambda_pop`** is a learned function of multi-scale decayed counts of
  `(r, o)` and of `o`, for all entities, with no top-k cut.
- **`lambda_struct`** is snapshot evolution with a ConvTransE decoder. It is
  prior work (RE-GCN, DiMNet) and is only the backbone.

Why the path intensity: `diagnose_paths.py` on ICEWS18 finds that 54 % of
cold answers are a partner of one of the subject's 96 most recent partners,
inside a set of about 555 entities, which is 2.4 % of all entities. A
hand-built path counter combined with popularity reaches MRR 4.51 on the
cold stratum against 2.86 for popularity alone.

Relation to prior work: the type-to-type counter above is close to the
length-1 rules of TLogic and to the rule confidences of CountTRuCoLa, and
the path intensity to their length-2 rules. LogCL and HisRES reach the
subject's wider history with a query-specific global graph. ECHO reads the
same evidence as event streams and composes them along paths.

## Run

```bash
./get_data.sh
python verify.py --dataset ICEWS14s
python diagnose_dyad.py --dataset ICEWS18
python diagnose_paths.py --dataset ICEWS18
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

Ablations: `--no_path`, `--no_proto`, `--no_ctx`, `--compete`, `--no_type` (stream keeps times,
loses relation types), `--no_stream` (statistics only), `--no_dyad`,
`--no_pop`, `--no_struct`.

`verify.py` checks that deleting every event at or after `t` changes nothing
the model reads at `t`, and compares the vectorised support builder with a
brute-force loop.

## Protocol

Validation MRR selects the checkpoint. Test reports raw and time-aware
filtered ranks. Ties resolve to their average rank. Time is the snapshot
index, so one step is one snapshot on every dataset. The history before a
test timestamp is the true history, which is the RE-GCN setting.

## Results

Final code, three seeds (1, 2, 3), time-aware filtered, ties averaged, x100.
Mean +- standard deviation over seeds. Published numbers are the papers'
own; LogCL was re-run in this protocol by the authors of this repository
and lands at about 36.0 on ICEWS18.

| dataset | MRR | H@1 | H@3 | H@10 | strongest published (MRR / H@1) |
|---|---|---|---|---|---|
| YAGO | **91.82** +- 0.20 | **90.57** +- 0.44 | 92.91 | 93.23 | DaeMon 91.59 / 90.03 |
| WIKI | **83.41** +- 0.12 | **80.44** +- 0.26 | **86.04** | 87.39 | CognTKE 83.21; DaeMon 82.38 / 78.26 |
| GDELT | 27.01 +- 0.03 | **18.05** +- 0.03 | 29.81 | 44.60 | CID-TKG 27.41 / 17.76 (no code); HisRES 26.58 / 16.90 |
| ICEWS18 | 36.29 +- 0.04 | 26.10 +- 0.04 | 40.96 | 56.06 | LogCL 35.67 (36.0 re-run); CID-TKG 38.88 (no code) |
| ICEWS14s | 47.18 +- 0.04 | 37.19 +- 0.11 | 52.43 | 66.16 | LogCL 48.87; DiMNet 45.72 |

What this does and does not say. On YAGO and WIKI ECHO is above every
published number we found on MRR and H@1. On GDELT it has the best H@1 and
is 0.4 MRR behind a preprint without code. On ICEWS18 and ICEWS14s it is
above every model with released code that we could run in this protocol,
and behind HisRES, CID-TKG and CHE-TKG, none of which has code.

### Strata (ICEWS18, three seeds)

| stratum | share | MRR | H@1 | H@10 |
|---|---|---|---|---|
| cold_far | 17.8 % | 1.84 | 0.64 | 3.18 |
| cold_2hop | 14.0 % | 8.93 | 3.13 | 19.46 |
| dyad_only | 19.2 % | 29.71 | 17.21 | 56.72 |
| blocked | 25.7 % | 34.99 | 16.82 | 72.96 |
| clean | 23.4 % | 85.64 | 76.65 | 99.02 |

### Ablations, YAGO (seed 1; full model is the 3-seed mean)

| variant | MRR | H@1 | delta MRR |
|---|---|---|---|
| full | 91.82 | 90.57 | |
| no stream (statistics only) | 91.83 | 90.77 | 0.00 |
| no relation types in the stream | 91.79 | 90.58 | -0.03 |
| no popularity field | 91.81 | 90.66 | -0.01 |
| no path intensity | 91.59 | 90.41 | -0.23 |
| no candidate context | 91.61 | 90.31 | -0.21 |
| no prototype | 91.54 | 90.21 | -0.28 |
| no structural branch | 90.97 | 89.14 | -0.85 |
| no dyad branch | 67.18 | 61.37 | -24.64 |

On YAGO the eight dyad statistics carry the dyad branch on their own; the
sequence encoder and the relation types add nothing there, as the
diagnostic predicted (dyad_only is 0.1 % of YAGO). The dyad branch as a
whole is worth 24.6 MRR. ICEWS18 ablations are running.

### Earlier development runs (ICEWS18, seed 42)

| variant | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| dyad + pop + struct | 35.80 | 25.70 | 40.40 | 55.38 |
| + path | 36.26 | 26.02 | 40.94 | 56.05 |
| + path + competition (opt-in now) | 36.27 | 26.14 | 40.93 | 55.90 |
| + path + prototype | 36.25 | 26.08 | 40.91 | 55.88 |
| + path + prototype + context (final) | 36.32 | 26.14 | 40.95 | 56.14 |
| structural branch alone | 30.27 | 20.54 | 33.95 | 49.59 |

Ten training recipes (schedule, dropout, weight decay, learning rate,
auxiliary weight) all land between 36.0 and 36.3 on ICEWS18 and all peak at
epoch 8-10: the plateau is not a regularisation problem.

Data is not tracked in git. Put each dataset in `data/<NAME>/` with
`train.txt`, `valid.txt` and `test.txt`, or run `./get_data.sh`.
