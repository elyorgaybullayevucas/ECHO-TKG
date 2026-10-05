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

Ablations: `--no_path`, `--no_compete`, `--no_type` (stream keeps times,
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

## Status

Measured so far, ICEWS18, time-aware filtered, one seed (42):

| variant | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| full | 36.27 | 26.14 | 40.93 | 55.90 |
| without competition | 36.26 | 26.02 | 40.94 | 56.05 |
| without the path intensity | 35.84 | 25.88 | 40.47 | 55.03 |

The path intensity is worth +0.43 MRR and lands where it was aimed: on
`cold_2hop` (14 % of queries) MRR goes from 6.21 to 8.52 and H@10 from 13.43
to 19.04. The competition layer is a null result on one seed: +1.2 MRR on
`clean`, -1.5 on `dyad_only`, nothing overall.

Every run peaked at epoch 8 to 12 of 40 with the learning rate still near
its maximum. `run_sweep.sh` tests the training recipe for that reason.

No claim over published numbers is made until three seeds are in.

Data is not tracked in git. Put each dataset in `data/<NAME>/` with
`train.txt`, `valid.txt` and `test.txt`, or run `./get_data.sh`.
