import os
import yaml
from stg.datasets import valid_dataset
from stg.gan.loss import valid_loss
from schema import Schema, SchemaError, Optional, And, Or


config_schema = Schema({
    "project": str,
    "name": str,
    "out-dir": os.path.exists,
    "data-dir": os.path.exists,
    "fid-stats-path": os.path.exists,
    "fixed-noise": Or(And(str, os.path.exists), int),
    "test-noise": os.path.exists,
    Optional("compute-fid"): bool,
    Optional("device", default="cpu"): str,
    Optional("num-workers", default=0): int,
    Optional("num-runs", default=1): int,
    Optional("step-1-seeds"): [int],
    Optional("step-2-seeds"): [int],
    "dataset": {
        "name": And(str, valid_dataset),
        Optional("binary"): {"pos": int, "neg": int}
    },
    "model": {
        "z_dim": int,
        "g_filter_dim": int,
        "d_filter_dim": int,
        "g_num_blocks": int,
        "d_num_blocks": int,
        "loss": Or({
            "name": "wgan-gp",
            "args": {
                "lambda": int,
            }
        }, {
            "name": "ns"
        })
    },
    "optimizer": {
        "lr": float,
        "beta1": float,
        "beta2": float,
    },
    "train": {
        "step-1": Or(And(str, os.path.exists), {
            "epochs": int,
            "batch-size": int,
            Optional("early-stop"): {
                "criteria": int,
            }
        }),
        "step-2": {
            Optional("early-stop"): {
                "criteria": int,
            },
            "epochs": int,
            "batch-size": int,
            "classifier": [And(str, os.path.exists)],
            "weight": [And(Or(int, float), lambda n: 0 <= n <= 1)]
        }
    }
})


def read_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    try:
        config_schema.validate(config)
    except SchemaError as se:
        raise se

    if "run-seeds" in config and len(config["run-seeds"]) != config["num-runs"]:
        print("Number of seeds must be equal to number of runs")
        exit(-1)

    if "run-seeds" in config["train"]["step-2"] and \
            len(config["train"]["step-2"]["run-seeds"]) != config["num-runs"]:
        print("Number of mod_gan seeds must be equal to number of runs")
        exit(-1)

    return config