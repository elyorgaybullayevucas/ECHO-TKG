"""ECHO configuration."""
import argparse
import os
from dataclasses import dataclass, fields
from typing import Tuple

_C = dict(
    embed_dim=200, gcn_layers=2, conv_channels=50, hist_len=10,
    stream_dim=64, stream_layers=2, stream_heads=4, stream_len=10,
    dyad_support=96, triple_support=64, horizon=0,
    path_support=16, path_dim=32,
    pop_dim=16, dropout=0.2, bias_init=-2.0, struct_aux=0.3,
    lr=1e-3, weight_decay=1e-5, grad_clip=1.0, label_smoothing=0.1,
    warmup_ratio=0.05, epochs=40, patience=8,
    query_chunk=1024, cand_budget=60000,
)

DATASETS = {
    "ICEWS14s": dict(_C, dropout=0.25),
    "ICEWS18": dict(_C, dropout=0.25),
    "GDELT":   dict(_C, hist_len=5, dropout=0.25, epochs=30, patience=6,
                    dyad_support=64, triple_support=48, stream_len=8),
    "YAGO":    dict(_C, dropout=0.15, struct_aux=0.0, dyad_support=48,
                    stream_len=8),
    "WIKI":    dict(_C, dropout=0.15, struct_aux=0.0, dyad_support=48,
                    stream_len=8),
}


@dataclass
class EchoConfig:
    dataset: str = "ICEWS18"
    data_dir: str = "./data"
    embed_dim: int = 200
    gcn_layers: int = 2
    conv_channels: int = 50
    hist_len: int = 10
    stream_dim: int = 64
    stream_layers: int = 2
    stream_heads: int = 4
    stream_len: int = 10
    dyad_support: int = 96
    triple_support: int = 64
    path_support: int = 16      # second-hop edges per first-hop candidate
    path_dim: int = 32
    horizon: int = 0            # snapshots of history the supports may use; 0 = all
    pop_dim: int = 16
    dropout: float = 0.2
    bias_init: float = -2.0
    struct_aux: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    warmup_ratio: float = 0.05
    epochs: int = 40
    patience: int = 8
    query_chunk: int = 1024
    cand_budget: int = 60000    # max (query, candidate) pairs per chunk
    max_snapshots: int = 0      # debugging: train on this many timestamps
    # ablations
    no_stream: bool = False     # dyad branch sees the 8 statistics only
    no_type: bool = False       # stream keeps times, loses relation types
    no_dyad: bool = False       # no dyad branch at all
    no_path: bool = False       # no two-hop path intensity
    no_compete: bool = False    # candidates scored independently
    no_pop: bool = False        # no popularity field
    no_struct: bool = False     # no structural branch
    eval_only: bool = False
    no_amp: bool = False        # full precision on the GPU
    force_amp: bool = False     # testing only: bf16 autocast on CPU
    num_workers: int = -1
    hits_at: Tuple = (1, 3, 10)
    seed: int = 42
    device: str = "auto"
    gpu: int = 0                # -1 = the GPU with the most free memory
    tag: str = ""
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


def parse_args(argv=None):
    p = argparse.ArgumentParser("ECHO")
    p.add_argument("--dataset", default="ICEWS18", choices=list(DATASETS))
    for f in fields(EchoConfig):
        if f.name == "dataset" or f.name == "hits_at":
            continue
        if f.type is bool:
            p.add_argument(f"--{f.name}", action="store_true")
        else:
            p.add_argument(f"--{f.name}", type=f.type, default=None)
    a = p.parse_args(argv)
    base = dict(DATASETS[a.dataset])
    for k, v in vars(a).items():
        if v is not None and v is not False:
            base[k] = v
    cfg = EchoConfig(**base)
    if cfg.num_workers < 0:
        # fork is cheap on Linux; Windows would re-import and copy the index
        cfg.num_workers = 0 if os.name == "nt" else min(6, os.cpu_count() or 1)
    return cfg
