import argparse
from dataclasses import dataclass, field
from typing import Literal

from ..data_handler import DATASET_NAME_LIST

SEED = 42  # Default seed for reproducibility


@dataclass
class GridSearchArgs:
    dataset_names: list[str] = field(default_factory=lambda: DATASET_NAME_LIST)
    experiment_name: str = "test"
    data_source: Literal["synthetic", "local", "public"] = "local"
    n_splits: int = 0
    n_runs: int = 1
    seed: int = SEED
    classification: bool = False
    max_train_data_size: int = 0
    max_workers: int = 0

    @classmethod
    def parse(cls) -> "GridSearchArgs":
        parser = argparse.ArgumentParser(
            prog="grid-search",
            description="Run Rutile grid search",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "-d", "--dataset_names", nargs="+", default=cls().dataset_names
        )
        parser.add_argument("-e", "--experiment_name", default=cls().experiment_name)
        parser.add_argument(
            "-ds",
            "--data_source",
            choices=["synthetic", "local", "public"],
            default=cls().data_source,
        )
        parser.add_argument("--n_splits", type=int, default=cls().n_splits)
        parser.add_argument("-n", "--n_runs", type=int, default=cls().n_runs)
        parser.add_argument("--seed", type=int, default=cls().seed)
        parser.add_argument("-c", "--classification", action="store_true")
        parser.add_argument(
            "--max_train_data_size", type=int, default=cls().max_train_data_size
        )
        parser.add_argument("--max_workers", type=int, default=cls().max_workers)
        return cls(**vars(parser.parse_args()))
