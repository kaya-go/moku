# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#     "torch>=2.13.0,<3",
#     "transformers>=5.16.1,<6",
#     "datasets>=5.0.1,<6",
#     "huggingface_hub>=1.32.0,<2",
#     "albumentations>=2.0.8,<3",
#     "numpy>=2.5.3,<3",
#     "scipy>=1.18.1,<2",
#     "scikit-learn>=1.9.1,<2",
#     "pandas>=3.0.6,<4",
#     "pillow>=12.3.0,<13",
#     "torchmetrics>=1.9.0,<2",
#     "pycocotools>=2.0.11,<3",
#     "faster-coco-eval>=1.8.0,<2",
# ]
# ///
"""Train a moku detector (see ``moku.training.engine`` for the recipe).

The training code lives in the ``moku`` package. On HF Jobs, ``src/`` is
mounted at ``/moku-src`` and the runs bucket at ``/runs``; ``moku train launch``
builds that command. Locally, the ``moku`` package of the pixi environment is used:

    pixi run python scripts/train.py --run-name smoke --limit-train 32 --limit-eval 8 --epochs 2

Every ``TrainConfig`` field is a flag (``--no-aug-epochs 8``, ``--model dfine-s``, ...).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

MOKU_SRC = "/moku-src"
if os.path.isdir(MOKU_SRC):
    sys.path.insert(0, MOKU_SRC)

from moku.training.engine import TrainConfig, train  # noqa: E402

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
