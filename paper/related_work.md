# Related work: verified notes

Every entry below was checked against the paper's own abstract or full text
(arXiv HTML where available) on 2026-10-06. "Mechanism" is a faithful
paraphrase in our words; the few short phrases in quotation marks are the
paper's own wording and may be quoted as such. Numbers are the papers' own,
time-aware filtered unless marked, and are NOT comparable across papers
(see §7). "vs ECHO" says what the method sees that ECHO does not, or the
reverse.

Keys match `references.bib`.

---

## 1. Recurrence and copy mechanisms

**CyGNet** — `zhu2021cygnet` — Zhu, Chen, Fan, Cheng, Zhan. *Learning from
History: Modeling Temporal Knowledge Graphs with Sequential Copy-Generation
Networks.* AAAI 2021. arXiv:2012.08492.
Mechanism: a "time-aware copy-generation mechanism" with two modes: a copy
mode that scores entities that already appeared as answers of the same
(s, r) in the past, and a generation mode over the whole vocabulary; the
abstract motivates this by the observation that many facts recur.
vs ECHO: the copy vocabulary is the (s, r) history only. Our dyad_only
stratum (the answer interacted with s under some other relation) is outside
it by construction.

**CENET** — `xu2023cenet` — Xu, Ou, Xu, Fu. *Temporal Knowledge Graph
Reasoning with Historical Contrastive Learning.* AAAI 2023. arXiv:2211.10904.
Mechanism: learns "both the historical and non-historical dependency",
trains the query representation contrastively to decide which of the two the
answer follows, and uses a binary classifier to produce a mask over
candidates at inference.
vs ECHO: same (s, r)-history notion of "historical". The mask is a hard
gate; ECHO has no gate, intensities add.

**Recurrency baseline** — `gastinger2024history` — Gastinger, Meilicke,
Errica, Sztyler, Schuelke, Stuckenschmidt. *History repeats Itself: A
Baseline for Temporal Knowledge Graph Forecasting.* IJCAI 2024.
arXiv:2404.16726.
Mechanism: predicts by "recurring facts" with a strict and a relaxed
recurrency score and a handful of hyper-parameters, no iterative training.
Finding: against eleven methods on five datasets it is competitive or better
on three. Reported (own table): ICEWS18 MRR 28.7, GDELT 24.5, WIKI 81.5,
YAGO 90.9.
vs ECHO: our diagnose_dyad.py "recurrence" counter is this idea (ICEWS18
28.0 on a 30k sample); allowing the whole dyad raises the untrained counter
to 31.8. This is the direct motivation of the dyad.

**TiRGN** — `li2022tirgn` — Li, Sun, Zhao. *TiRGN: Time-Guided Recurrent
Graph Network with Local-Global Historical Patterns for Temporal Knowledge
Graph Reasoning.* IJCAI 2022.
Mechanism: "a local recurrent graph encoder network to model the historical
dependency of events at adjacent timestamps", a global history encoder for
repeated facts, and a time-guided decoder with periodicity. Reported ICEWS18
33.66 / 23.19 (as quoted by later papers).
vs ECHO: the global encoder is again (s, r)-history; the periodicity is in
the decoder, ours is in the time encoding of the stream.

## 2. Structural evolution over snapshots

**RE-NET** — `jin2020renet` — Jin, Qu, Jin, Ren. *Recurrent Event Network:
Autoregressive Structure Inference over Temporal Knowledge Graphs.*
EMNLP 2020. arXiv:1904.05530.
Mechanism: an autoregressive model with a recurrent event encoder over past
graphs and a neighbourhood aggregator for concurrent events.

**RE-GCN** — `li2021regcn` — Li, Jin, Li, Guan, Guo, Shen, Wang, Cheng.
*Temporal Knowledge Graph Reasoning Based on Evolutional Representation
Learning.* SIGIR 2021. arXiv:2104.10353.
Mechanism: relation-aware GCN per snapshot, a gated recurrent unit across
snapshots, and a static-graph constraint on entity representations; the
abstract reports up to 11.46 % MRR improvement and up to 82x speed-up over
prior work. Reported ICEWS18 30.58 / 21.01, GDELT 19.64 / 12.42 (DiMNet
table).
vs ECHO: our structural branch is this family (we measured it alone at
ICEWS18 MRR 30.27). RE-GCN's `utils.py` is the de-facto evaluation
reference. Its static-graph constraint uses extra files (entity types) that
we do not use.

**CEN** — `li2022cen` — Li, Guan, Jin, Peng, Lyu, Zhu, Bai, Li, Guo, Cheng.
*Complex Evolutional Pattern Learning for Temporal Knowledge Graph Reasoning.*
ACL 2022 (short).
Mechanism: "a length-aware Convolutional Neural Network (CNN) to handle
evolutional patterns of different lengths via an easy-to-difficult
curriculum learning strategy", plus an online-learning setting in which the
model keeps training as new snapshots arrive. Reported ICEWS18 30.84 / 21.23.
Note: the online setting is a different protocol; numbers under it must be
reported separately.

**HiSMatch** — `li2022hismatch` — Li, Hou, Guan, Jin, Peng, Bai, Lyu, Li,
Guo, Cheng. *HiSMatch: Historical Structure Matching based Temporal Knowledge
Graph Reasoning.* EMNLP 2022 Findings. arXiv:2210.09708.
Mechanism: reasoning as "a matching task between a query and candidate
entities based on their historical structures", with one encoder for the
query's history, one for each candidate's history, and a background
encoder. Reported ICEWS18 33.99 / 23.91 (NADEx table).
vs ECHO: the closest precedent for our candidate-context input (the
candidate's own recent typed edges). HiSMatch encodes structures per
timestamp; we read the candidate's edge list directly.

**DiMNet** — `dong2025dimnet` — Dong, Qiao, Ning, Hao, Du, Wang, Zhou.
*Disentangled Multi-span Evolutionary Network against Temporal Knowledge
Graph Reasoning.* ACL 2025 Findings. arXiv:2505.14020.
Mechanism: multi-span evolution with an internal interaction mechanism
between subgraphs during evolution, and a disentanglement component that
separates nodes' active and stable features. Reported ICEWS18 34.13 / 23.29,
GDELT 21.93 / 14.03; its own table is the source of the Group A baseline
numbers.

## 3. Global history graphs (the current strongest family)

**LogCL** — `chen2024logcl` — Chen, Wan, Wu, Zhao, Cheng, Li, Lin.
*Local-Global History-aware Contrastive Learning for Temporal Knowledge Graph
Reasoning.* ICDE 2024. arXiv:2312.01601.
Mechanism: an entity-aware attention over a local history (recent snapshots)
and a global history subgraph built per query from (i) the one-hop
historical facts of the query subject and (ii) the one-hop facts of the
objects that answered (s, r) before; four historical query contrast patterns
regularise the fusion. Testing runs in two phases (original queries, then
inverse queries) so the inverse set cannot leak answers. Reported ICEWS18
35.67 / 24.53 / 40.32 / 57.74, GDELT 23.75 / 14.64. Ablation (ICEWS14):
local-only 44.74, global-only 46.81, full 48.87. Code released.
Re-run under our protocol by us: ICEWS18 MRR about 36.0.
vs ECHO: LogCL's global subgraph contains the same facts our dyad streams
and second-hop edges read, but it processes them with message passing and
contextualises every entity in the subgraph; ECHO reads them as sequences
and scores candidates directly.

**HisRES** — `zhang2024hisres` — Zhang, Hui, Mu, Sun, Tian. *Historically
Relevant Event Structuring for Temporal Knowledge Graph Reasoning.* ICDE 2025
(per the authors); arXiv:2405.10621.
Mechanism: a multi-granularity evolutionary encoder over the most recent
snapshots (merging adjacent snapshots, span omega=2 by default) and a global
relevance encoder over all historical facts that share the query's (s, r),
fused by self-gating; two-phase propagation for raw and inverse queries.
Reported ICEWS18 37.69 / 26.46 / 42.75 / 59.70, GDELT 26.58 / 16.90. Ablation
without the global encoder: ICEWS18 31.55. **No code.**

**CID-TKG** — `lei2026cidtkg` — Lei, Zhu, Liang, Sun, Fang, Yin.
*CID-TKG: Collaborative Historical Invariance and Evolutionary Dynamics
Learning for Temporal Knowledge Graph Reasoning.* arXiv:2604.09600 (2026,
unrefereed at the time of writing).
Mechanism: a historical-invariance graph (all past facts sharing the query
relation, timestamps removed) and an evolutionary-dynamics graph (recent
facts retrieved through temporal logical rules), each with its own encoder,
view-specific relation decomposition, and InfoNCE alignment between the two
query views. Reported ICEWS18 38.88 / 27.66 / 43.78 / 61.16, GDELT 27.41 /
17.76 / 29.92 / 47.03. Embedding 200, history 3 snapshots. **No code.**

**CHE-TKG** — `lei2026chetkg` — same authors. *CHE-TKG: Collaborative
Historical Evidence and Evolutionary Dynamics Learning for Temporal Knowledge
Graph Reasoning.* arXiv:2605.04652 (2026, unrefereed).
Mechanism: as CID-TKG with the invariance graph replaced by all past facts of
the same (s, r). Reported ICEWS18 38.77 / 27.58 / 60.95 (H@10), GDELT 27.38 /
17.73 / 47.02; mean of three seeds. Ablation ICEWS18: without evolutionary
dynamics 36.88, without historical evidence 36.98, without contrastive
alignment 38.62. **No code.**
vs ECHO: the rule-retrieved recent facts are the multi-hop form of what our
dyad stream (length-1) and path intensity (length-2) read. The contrastive
alignment is worth 0.15 MRR in their own ablation.

**CognTKE** — `chen2025cogntke` — Chen, Wu, Wu, Zhang, Liao, Lin, Wan.
*CognTKE: A Cognitive Temporal Knowledge Extrapolation Framework.* AAAI 2025.
arXiv:2412.16557.
Mechanism: a "temporal cognitive relation directed graph" holding local and
global historical paths; a global one-hop reasoner (fast) and a local
multi-hop reasoner (slow), after the dual-process account of cognition.
Reported ICEWS18 35.24, WIKI 83.21 (own paper).

## 4. Paths, rules and explainable reasoning

**xERTE** — `han2021xerte` — Han, Chen, Ma, Tresp. *xERTE: Explainable
Reasoning on Temporal Knowledge Graphs for Forecasting Future Links.*
ICLR 2021. arXiv:2012.15537.
Mechanism: expands a query-relevant subgraph by iteratively sampling temporal
neighbours with temporal relational attention and a reverse representation
update, producing "human-understandable evidence". Reported ICEWS18 29.31 /
21.03, YAGO 84.19 / 80.09.

**TITer** — `sun2021titer` — Sun, Zhong, Ma, Han, He. *TimeTraveler:
Reinforcement Learning for Temporal Knowledge Graph Forecasting.* EMNLP 2021.
arXiv:2109.04101.
Mechanism: an agent walks over historical snapshots with relative time
encoding and a Dirichlet-based reward; "a novel representation method for
unseen entities" gives inductive ability. Reported ICEWS18 29.98 / 22.05,
YAGO 87.47 / 84.89.

**TLogic** — `liu2022tlogic` — Liu, Ma, Hildebrandt, Joblin, Tresp.
*TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal
Knowledge Graphs.* AAAI 2022. arXiv:2112.08025.
Mechanism: extracts "temporal logical rules via temporal random walks",
applies them with time-consistent groundings, and transfers to inductive
settings with shared vocabularies. Reported ICEWS14 43.04 / 33.56 / 61.23,
ICEWS18 29.82 (own paper).
vs ECHO: our untrained typed-dyad counter (ICEWS14s 42.72 / 33.73 / 60.39)
behaves like TLogic's length-1 rules; the path intensity is the learned
form of its length-2 rules.

**CountTRuCoLa** — `gastinger2026counttrucola` — Gastinger, Meilicke,
Stuckenschmidt. *CountTRuCoLa: Rule Learning for Interpretable Temporal
Knowledge Graph Forecasting.* ISWC 2026 (per the arXiv page).
arXiv:2509.09474.
Mechanism: "four simple rule types, including temporal rules with confidence
functions that combine both recency and frequency"; every prediction is
attributable to rules and their groundings. Reported ICEWS18 32.8, GDELT 23.8,
WIKI 82.7, YAGO 90.9.
vs ECHO: closest symbolic relative. Its confidence functions are monotone in
recency; ECHO's time encoding is not constrained to be.

**DaeMon** — `dong2023daemon` — Dong, Ning, Wang, Qiao, Wang, Zhou, Fu.
*Adaptive Path-Memory Network for Temporal Knowledge Graph Reasoning.*
IJCAI 2023. arXiv:2304.12604.
Mechanism: a path memory that tracks temporal paths between the query
subject and candidate objects across adjacent timestamps; the paper states
the model "models the historical information without depending on entity
representation". Reported WIKI 82.38 / 78.26 / 86.03 / 88.01, YAGO 91.59 /
90.03 / 93.00 / 93.34, ICEWS18 31.85 / 22.67, GDELT 20.73 / 13.65.

**TiPNN** — `dong2024tipnn` — Dong, Wang, Xiao, Ning, Wang, Zhou. *Temporal
Inductive Path Neural Network for Temporal Knowledge Graph Reasoning.*
Artificial Intelligence, 2024. arXiv:2309.03251.
Mechanism: a unified history temporal graph and "query-aware temporal paths
on a history temporal graph", entity-independent and therefore inductive.
Reported ICEWS18 32.17 / 22.74, GDELT 21.17 / 14.03.

## 5. Point processes

**Know-Evolve** — `trivedi2017knowevolve` — Trivedi, Dai, Wang, Song.
*Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs.*
ICML 2017. arXiv:1705.05742.
Mechanism: each fact is "a multivariate point process whose intensity
function is modulated by the score for that fact computed based on the
learned entity embeddings", with entity embeddings evolving in continuous
time.
vs ECHO: the first point-process view of TKGs; the intensity is a function
of embeddings, not of the pair's event stream.

**GHNN** — `han2020ghnn` — Han, Ma, Wang, Günnemann, Tresp. *Graph Hawkes
Neural Network for Forecasting on Temporal Knowledge Graphs.* AKBC 2020.
arXiv:2003.13432.
Mechanism: extends the neural Hawkes process to graph sequences; the authors
argue that treating every possible link as its own event-type dimension is
infeasible because the number of types explodes.
Correction to an earlier note of ours: GHNN does not implement a per-pair
Hawkes baseline; it gives the combinatorial argument against one. ECHO's
per-dyad stream is feasible precisely because only dyads with history are
ever instantiated (support of 96 + 64 candidates per query).

**GAttNHP** — `tian2026gattnhp` — Tian, Yu, Dai, Tang, Zhu. *GAttNHP: Group
Attention Neural Hawkes Process for Extrapolation Reasoning in Temporal
Knowledge Graphs.* arXiv:2607.14733 (2026, unrefereed).
Mechanism: a self-attention encoder over subject–relation chains as
continuous-time point processes, soft grouping that turns learnable Hawkes
priors into cross-attention masks, and a non-crossing quantile head for
inter-arrival times. Reports **raw** metrics (ICEWS18 38.63 / 28.25), which
are not comparable to filtered numbers.
vs ECHO: the chain is (s, r); ours is the dyad (s, o) under every relation.

## 6. Diffusion models

**DiffuTKG** — `cai2024diffutkg` — Cai, Liu, Gan, Li, Liu, Lin, Luo, Yang. *Predicting the
Unpredictable: Uncertainty-Aware Reasoning over Temporal Knowledge Graphs via
Diffusion Process.* ACL 2024 Findings.
Mechanism: future-fact prediction as sequence denoising: Gaussian noise
corrupts the target fact, a Transformer conditional denoiser restores it,
and an uncertainty regulariser counters the bias towards frequent facts.
Reported ICEWS18 35.65 / 25.19 (FreqDiff table) or 35.65 / 27.19 (NADEx
table) — the two later papers disagree on its H@1.

**NADEx** — `nadex2026` — Gan, He, Cai, Lin, Zhou, Liu. *Negative-Aware Diffusion Process for Temporal
Knowledge Graph Extrapolation.* EACL 2026 Findings. arXiv:2602.08815.
Mechanism: subject-centric histories encoded as sequences; the query object
is perturbed and reconstructed by a Transformer denoiser; batch-wise negative
prototypes enter the conditioning and a cosine-alignment term separates the
denoised embedding from negatives. Reported ICEWS18 36.84 / 27.58 / 45.12 /
60.58, GDELT 23.67 / 16.10. Code released.

**FreqDiff** — `freqdiff2026` — Gan, He, Lin, Jiang, Wang, Liu. *Denoising the Future: Context-Aware
Spectral Diffusion for Temporal Knowledge Graph Extrapolation.*
arXiv:2608.20804 (2026, unrefereed).
Mechanism: query-slot denoising with a dual-stream denoiser (temporal
Transformer + context-aware spectral calibration) and a frequency-domain
alignment loss. Reported ICEWS18 36.85 / 26.85 / 41.51 / 61.47, GDELT 25.04 /
17.64. Anonymous code link. Its table lists NADEx at ICEWS18 35.37, which
NADEx's own paper reports as 36.84.

## 7. Evaluation: protocol, shortcuts, strata

**Gastinger et al. 2023** — `gastinger2023apples` — Gastinger, Sztyler,
Sharma, Schuelke, Stuckenschmidt. *Comparing Apples and Oranges? On the
Evaluation of Methods for Temporal Knowledge Graph Forecasting.*
ECML PKDD 2023. Code: github.com/nec-research/TKG-Forecasting-Evaluation.
Finding: inconsistent experimental procedures (filter settings, single- vs
multi-step prediction, dataset versions) strongly influence results and
distort comparisons; the paper gives a unified protocol and re-evaluates
state-of-the-art models under it.
Use: justifies re-running LogCL in our protocol and refusing to pool numbers
across papers.

**Towards Better Evolution Modeling** — `zhang2026evolution` — Zhang, Li,
Wang, Shao, Cui, Li. arXiv:2602.08353 (2026, unrefereed).
Finding: a model that only counts co-occurrences, with no temporal
information, reaches Hits@10 above 0.9 on standard datasets; the authors list
dataset biases, over-simplified tasks, time-interval formatting and ignored
obsolescence, and release four bias-corrected datasets and two new tasks.

**Strikingness-Aware Evaluation** — `strikingness2026` — Huang, Zhang, Wei.
arXiv:2605.13153 (2026, unrefereed).
Finding: over 80 % of ICEWS test events recur from history; reweighting by
event rarity drops path- and rule-based methods (Recurrency, TLogic, TITer)
by 30–50 % and representation methods by under 30 %, LogCL the least.

**Distribution Shifts** — `shift2026` — Özdemir, Gastinger, Kirchdorfer, Stuckenschmidt. *Temporal Knowledge Graph
Forecasting under Distribution Shifts: A Synthetic Evaluation.*
arXiv:2607.09232 (2026, unrefereed).
Finding: on a synthetic generator with recurrence, homophily and
periodicity, memory-based baselines match neural models when recurrence
dominates; structural breaks in homophily hurt DiMNet most (88 % to 48 % of
oracle).
Use: our stratified evaluation (cold_far / cold_2hop / dyad_only / blocked /
clean) is the real-data counterpart of probing mechanisms separately.

## 8. Emerging entities and online adaptation (different settings)

**TransFIR** — `transfir2026` — Zhao, He, Wu, Tang, Lu, Gan, Fu, Wang, Zhou. *Inductive Reasoning for Temporal Knowledge
Graphs with Emerging Entities.* ICLR 2026. arXiv:2604.10164. Interaction-aware
codebook for entities unseen in training. Evaluated in an emerging-entity
split, not the standard one.

**AdaTKG** — `adatkg2026` — Lee, Seo, Lee, Yoo, Kim, Lim, Kang, Choi, Lee, Ahn. *AdaTKG: Adaptive Memory for Temporal Knowledge
Graph Reasoning.* arXiv:2605.07121 (2026). Per-entity memory updated by a
learnable EMA with one shared scalar; gains of 4–24 % MRR on emerging
entities. Code released.

**HiTS-CL** — `hitscl2026` — Liu, Liu, Zuo, Zhao, Fu, Zhang, Zhuang, Chen, Li. *A Continual Learning Framework for
Long-Horizon Temporal Knowledge Graph Extrapolation.* arXiv:2609.36559
(2026). Snapshot-by-snapshot continual fine-tuning with multi-teacher
distillation; LogCL+HiTS-CL reaches ICEWS18 37.56 and GDELT 28.54 **under
online updates**, a different protocol from ours.

Our cold strata are not the emerging-entity problem: both entities are
known, they have simply never interacted.

---

## How to use these notes

- A sentence in the paper may state what a method does only if it matches
  the "Mechanism" line here. If the paper is later re-read and the line is
  wrong, fix it here first.
- Numbers go in the results table only with their source paper and protocol
  named. Group A (DiMNet table), Group B (DaeMon table) and "own paper" must
  not be mixed in one column without saying so.
- Quotations: at most the short phrases marked here, in quotation marks,
  with the citation. Everything else is paraphrased.
