# region Imports
import itertools
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from tqdm import tqdm

RUTILE_AI_PATH = Path("/app/dev/rutile-ai")
GRID_SEARCH_CONFIG_PATH = RUTILE_AI_PATH / "data/configs/gs"
sys.path.append(str(RUTILE_AI_PATH))


from rutile_ai import AnomalyPipeline, ClassifierPipeline, DataHandler
from rutile_ai.module import Module
from rutile_ai.utils import GridSearchArgs

# Disable mlflow logging
logging.getLogger("mlflow").setLevel(logging.WARNING)
# endregion Imports


if __name__ == "__main__":
    # region Initialization
    args = GridSearchArgs.parse()
    cross_validation_mode = args.n_splits > 0
    config_path = GRID_SEARCH_CONFIG_PATH / args.experiment_name
    grid_dict = json.load(
        open(
            config_path / "grid_dict.json",
            "r",
        )
    )
    fixed_config = json.load(
        open(
            config_path / "fixed_config.json",
            "r",
        )
    )
    grid = Module.flatten_dict(grid_dict)
    grid_combinations = [
        dict(zip(grid.keys(), combination))
        for combination in itertools.product(*grid.values())
    ]

    if args.classification:
        PipelineClass = ClassifierPipeline
    else:
        PipelineClass = AnomalyPipeline
    # endregion Initialization
    for dataset_name in args.dataset_names:
        try:
            # region MLFlow and experiment setup
            print(f"RUNNING GRID SEARCH FOR {dataset_name}")
            experiment_path = Path(
                RUTILE_AI_PATH
                / f"data/pipelines/{dataset_name}/gs_{args.experiment_name}/"
            )
            full_experiment_identifier = (
                f"{dataset_name}_{args.experiment_name}_gridsearch"
            )

            mlflow.set_tracking_uri(str(os.getenv("MLFLOW_TRACKING_URI")))
            mlflow.set_experiment(full_experiment_identifier)
            # endregion MLFlow and experiment setup

            # region DataHandler
            match args.data_source:
                case "local":
                    data_handler = DataHandler.from_path(dataset_name)
                case "public":
                    data_handler = DataHandler.from_public_data(
                        dataset_name,
                        cross_validation_mode=cross_validation_mode,
                        max_train_data_size=args.max_train_data_size,
                    )
                case _:
                    raise ValueError(
                        f"Invalid data source: {args.data_source}. Choose'local', or 'public'."
                    )
            # endregion DataHandler

            # region Grid search
            experiment_start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            experiment_file_path = (
                experiment_path
                / f"{dataset_name}_experiment_results_{experiment_start_time}.csv"
            )
            experiment_data_dict_list = []

            for config_index, grid_point in enumerate(grid_combinations):
                print(f"Grid point {config_index+1}/{len(grid_combinations)}")
                grid_point_name = "_".join(map(str, grid_point.values()))
                grid_point_path = experiment_path / grid_point_name
                grid_point_path.mkdir(parents=True, exist_ok=True)

                # Save the grid point configuration
                grid_point_file_path = grid_point_path / "grid_point.json"
                with open(grid_point_file_path, "w") as f:
                    json.dump(grid_point, f, indent=4)

                # Update the fixed config with the grid point values
                for nested_key, value in grid_point.items():
                    Module.update_nested_dict(
                        fixed_config, nested_key.split("/"), value
                    )

                try:
                    # region Pipeline training and testing
                    start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    run_name = f"{grid_point_name}_{start_time}"

                    with mlflow.start_run(run_name=run_name):
                        pipeline = PipelineClass(
                            config_dict=fixed_config,
                            module_id=run_name,
                        )
                        pipeline.store(grid_point_path)
                        # RFE: Add recover option
                        # pipeline = PipelineClass.recover_latest(grid_point_path)

                        if cross_validation_mode:
                            pipeline.cross_validate(
                                data_handler=data_handler,
                                n_splits=args.n_splits,
                                n_repeats=args.n_runs,
                                seed=args.seed,  # NOTE: Fixed seed for sharing the same splits in cross-validation
                                max_workers=args.max_workers,
                            )
                        else:
                            # RFE: Add mlflow support
                            for run_index in tqdm(range(args.n_runs)):
                                pipeline.train(
                                    data_handler=data_handler, should_retrain=True
                                )
                                pipeline.test(data_handler=data_handler)

                        # endregion Pipeline training and testing

                        # region MLFlow logging
                        params = pipeline.get_parameters(reduced_format=True)
                        metrics = pipeline.get_metrics(reduced_format=True)

                        if cross_validation_mode:
                            params["n_splits"] = args.n_splits
                            params["n_repeats"] = args.n_runs
                            metrics = {
                                k: v
                                for k, v in metrics.items()
                                if k.startswith("cross_validation_avg/")
                                or k.startswith("cross_validation_std/")
                            }
                        else:
                            # Keep all weighted average metrics
                            metrics = {
                                k: v for k, v in metrics.items() if "weighted avg/" in k
                            }

                        mlflow.log_params(params)
                        mlflow.log_metrics(metrics)

                        image_path_dict = pipeline.get_image_path_dict()
                        for (
                            local_image_path,
                            remote_image_path,
                        ) in image_path_dict.items():
                            remote_path = remote_image_path.parent
                            mlflow.log_artifact(str(local_image_path), str(remote_path))
                        # endregion MLFlow logging

                        experiment_data_dict = {
                            "experiment_name": args.experiment_name,
                            "grid_point_name": grid_point_name,
                            "start_time": start_time,
                        }
                        experiment_data_dict.update(params)
                        experiment_data_dict.update(metrics)

                        experiment_data_dict_list.append(experiment_data_dict)

                except Exception as e:
                    logging.error(f"Error in grid point {grid_point_name}")
                    logging.error(f"Exception: {e}")
                    continue
                # Save the experiment data to a CSV file
                experiment_df = pd.DataFrame(experiment_data_dict_list)
                experiment_df.to_csv(
                    experiment_file_path,
                    index=False,
                )

            print(f"Grid search completed correctly.")
        # endregion Grid search
        except Exception as e:
            logging.error(f"Error processing dataset {dataset_name}: {e}")
            continue
