"""Train a moku detector (see ``moku.training.engine`` for the recipe).

Runs in the pixi environment: ``default`` locally, ``cuda`` on HF Jobs, where
``moku train launch`` runs this script in the pixi image with the runs bucket at ``/runs``.

    pixi run python scripts/train.py --run-name smoke --limit-train 32 --limit-eval 8 --epochs 2

Every ``TrainConfig`` field is a flag (``--no-aug-epochs 8``, ``--model dfine-s``, ...).
"""

from __future__ import annotations

import argparse
import dataclasses
import os

from moku.training.engine import TrainConfig, train

_TYPES = {"int": int, "float": float, "str": str}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for field in dataclasses.fields(TrainConfig):
        flag = "--" + field.name.replace("_", "-")
        if field.type == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
        elif field.default is dataclasses.MISSING:
            parser.add_argument(flag, type=_TYPES[field.type], required=True)
        else:
            parser.add_argument(flag, type=_TYPES[field.type.split(" | ")[0]], default=field.default)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args(), extra_config={"git": os.environ.get("MOKU_GIT"), "job_id": os.environ.get("JOB_ID")})
