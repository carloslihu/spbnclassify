# RFE: Join all the classes in a single file
import pandas as pd

from ....data_handler import TRUE_CLASS_LABEL
from ..thresholder import Thresholder


class MaxProbabilityThresholder(Thresholder):
    true_label = TRUE_CLASS_LABEL
    prediction_label = "predicted_label"

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adjusts the threshold based on the data.

        Args:
            test_name (str): Name of the test dataset part to evaluate the models with
            data (pd.DataFrame): Data to evaluate.
        """
        # RFE: que exista un diccionario self.parameters?
        return self.transform_batch(data)

    def _thresholding_function(self, score: pd.DataFrame | pd.Series) -> pd.Series:
        # TODO: que sea el pipeline el que gestione si se usa o no el "anomaly_score"
        # RFE: lo más elegante es dejar de usar el nombre Thresholder para este tipo de módulos de clasificación -> Son más bien un decisor/comparador incluso ClassifierThresholder vs AnomalyThresholder blablabla
        predicted_series = pd.Series(
            score.drop("anomaly_score", axis=1, errors="ignore")
            .idxmax(axis=1)
            .str.replace("_score", ""),
        ).rename(self.prediction_label)
        return predicted_series
