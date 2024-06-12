import os
from schema import Schema, Optional, And, Or
from src.datasets import valid_dataset

CONFIG_SCHEMA_GASTEN = Schema({
    "project": str,
    "name": str,
    "out-dir": os.path.exists,
    "data-dir": os.path.exists,
    Optional("fid-stats-path"): os.path.exists,
    "fixed-noise": Or(And(str, os.path.exists), int),
    "test-noise": os.path.exists,
    Optional("compute-fid"): bool,
    Optional("device", default="cpu"): str,
    Optional("num-workers", default=0): int,
    Optional("num-runs", default=1): int,
    Optional("step-1-seeds"): [int],
    Optional("step-2-seeds"): [int],
    Optional("dataset"): {
        "name": And(str, valid_dataset),
        Optional("binary"): {"pos": int, "neg": int}
    },
    "model": {
        Optional("z_dim"): int,
        "architecture": Or({
            "name": "dcgan",
            Optional("g_filter_dim"): int,
            Optional("d_filter_dim"): int,
            Optional("g_num_blocks"): int,
            Optional("d_num_blocks"): int,
        }, {
            "name": "dcgan-v2",
            "g_filter_dim": int,
            "d_filter_dim": int,
            "g_num_blocks": int,
            "d_num_blocks": int,
        }, {
            "name": "resnet",
            "g_filter_dim": int,
            "d_filter_dim": int,
        }),
        "loss": Or({
            "name": "wgan-gp",
            "args": {
                "lambda": int,
            }
        }, {
            "name": "ns"
        })
    },
    Optional("optimizer"): {
        "lr": float,
        "beta1": Or(float, int),
        "beta2": Or(float, int),
    },
    "train": {
        Optional("step-1"): Or(And(str, os.path.exists), {
            "epochs": int,
            "checkpoint-every": int,
            Optional("batch-size"): int,
            "disc-iters": int,
            Optional("early-stop"): {
                "criteria": int,
            }
        }),
        Optional("step-2"): {
            Optional("step-1-epochs", default="best"): [Or(int, "best", "last")],
            Optional("early-stop"): {
                "criteria": int,
            },
            Optional("epochs"): int,
            Optional("checkpoint-every"): int,
            "batch-size": int,
            "disc-iters": int,
            Optional("classifier"): [And(str, os.path.exists)],
            Optional("weight"): [Or(int, float, "mgda", "mgda:norm")]
        }
    }
})

CONFIG_SCHEMA_CLUSTERING = Schema({
    "project": str,
    "name": str,
    Optional("device", default="cuda:0"): str,
    Optional("tag", default="draft"): str,
    "batch-size": int,
    "checkpoint": bool,
    "compute-fid": bool,
    Optional("dir"): {
        "data": os.path.exists,
        "clustering": os.path.exists,
        Optional("fid-stats"): str,
    },
    Optional("dataset"): {
        "name": And(str, valid_dataset),
        Optional("binary"): {"pos": int, "neg": int}
    },
    Optional("gasten"): {
        Optional("classifier"): [And(str, os.path.exists)],
        Optional("run-id"): str,
        Optional("weight"): int,
        "epoch": {
            "step-1": int,
            "step-2": int
        },
    },
    "clustering": {
        "z-dim": int,
        "fixed-noise": int,
        "acd": float,
        "n-iter": int,
        Optional("options"): [{
            "dim-reduction": Or("umap", "tsne"),
            "clustering": Or("hdbscan", "gmm")
        }]
    },
    "prototypes": {
        "type": [Or("medoid", "random")]
    }
})
