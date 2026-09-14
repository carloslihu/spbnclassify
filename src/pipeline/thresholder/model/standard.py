import pandas as pd

from ....data_handler import TRUE_ANOMALY_LABEL
from ..thresholder import Thresholder


class StandardThresholder(Thresholder):
    true_label = TRUE_ANOMALY_LABEL
    prediction_label = "binary_predicted_label"
    score_columns = ["anomaly_score"]

    def __init__(
        self,
        config_dict: dict,
        module_id: str,
        level: int,
    ):
        super().__init__(
            config_dict=config_dict,
            module_id=module_id,
            level=level,
        )
        if "threshold" in self.config_dict and self.config_dict["threshold"]:
            self.thresholder_parameters["threshold"] = self.config_dict["threshold"]

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adjusts the threshold based on the data.

        Args:
            test_name (str): Name of the test dataset part to evaluate the models with
            data (pd.DataFrame): Data to evaluate.
        """
        self.thresholder_parameters["threshold"] = self._calculate_threshold(
            data[self.score_columns]
        )
        return self.transform_batch(data)

    def _thresholding_function(self, score: pd.DataFrame | pd.Series) -> pd.Series:
        predicted_label = (score > self.thresholder_parameters["threshold"]).astype(int)
        predicted_label.rename(
            columns={self.score_columns[0]: self.prediction_label}, inplace=True
        )
        return predicted_label

    def _calculate_threshold(self, score: pd.DataFrame | pd.Series) -> pd.Series:
        mean = score.mean()
        std = score.std()

        # TODO: Uniformizar... hacer que sea n_sigmas
        if "num_sigmas" in self.config_dict:
            num_sigmas = self.config_dict["num_sigmas"]
        else:
            num_sigmas = 3

        threshold = mean + (num_sigmas * std)
        return pd.Series([threshold])
