from abc import ABC, abstractmethod
from typing import final

import pandas as pd

from ...module import Module


class Model(Module, ABC):
    """
    Model is an abstract base class for machine learning models within a pipeline. It inherits from Module and provides a standardized interface for training models.
    Attributes:
        trained (bool): Indicates whether the model has been trained.
    Args:
        config_dict (dict): Configuration parameters for the model.
        module_id (str): Unique identifier for the model module.
        level (int): Logging or verbosity level.
    Methods:
        learn(dataset_name: str, data: pd.DataFrame, should_retrain: bool = False) -> None:
            Trains the model on the provided dataset. If the model is already trained and should_retrain is False, training is skipped.
        _before_learn() -> None:
            Hook method called before training begins. Logs the start of training.
        _on_learn(dataset_name: str, data: pd.DataFrame) -> None:
            Abstract method to be implemented by subclasses, containing the core learning logic.
        _after_learn() -> None:
            Hook method called after training completes. Sets the trained flag to True.
    """

    def __init__(
        self,
        config_dict: dict,
        module_id: str,
        level: int,
    ) -> None:
        super().__init__(config_dict, module_id, level)
        # All models are initialized as untrained by default
        self.trained = False

    # region Learning methods
    @final
    def learn(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        should_retrain: bool = False,
        routing_column: str | None = None,
    ) -> None:
        # If the model should be retrained or is not trained yet, proceed with learning
        if should_retrain or not self.trained:
            self._before_learn()
            self._on_learn(dataset_name, data, routing_column)
            self._after_learn()
        else:
            self.log_note(
                self.level,
                f"Model {self.module_id} is already trained, skipping training",
            )

    def _before_learn(self) -> None:
        self.log_note(
            self.level,
            f"Training {self.module_id}",
        )

    @abstractmethod
    def _on_learn(
        self, dataset_name: str, data: pd.DataFrame, routing_column: str | None = None
    ) -> None:
        pass

    def _after_learn(self) -> None:
        self.trained = True

    # endregion Learning methods
