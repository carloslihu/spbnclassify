from pathlib import Path

import joblib
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ..single_model import ClassifierSingleModel

SKLEARN_MODELS = {
    "AdaBoostClassifier": AdaBoostClassifier,
    "CategoricalNB": CategoricalNB,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "GaussianNB": GaussianNB,
    "GaussianProcessClassifier": GaussianProcessClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "LogisticRegression": LogisticRegression,
    "MLPClassifier": MLPClassifier,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
    "RandomForestClassifier": RandomForestClassifier,
    "RBF": RBF,
    "SVC": SVC,
    "XGBClassifier": XGBClassifier,
}


class SklearnModel(ClassifierSingleModel):

    def __init__(self, config_dict: dict, module_id: str, level: int) -> None:
        """
        Initializes the engine wrapper with the given configuration.
        Args:
            config_dict (dict): Configuration dictionary containing the module type.
            module_id (str): Identifier for the module.
            level (int): Level of the module.
        Raises:
            ValueError: If the module type specified in the config_dict is not recognized.
        """
        module_type = config_dict["module_type"]

        if module_type in SKLEARN_MODELS:
            self.engine_model_class = SKLEARN_MODELS[module_type]
        else:
            raise ValueError(f"Unrecognized module type: {module_type}")
        super().__init__(config_dict, module_id, level)

    def save_module(
        self, module_path: Path = Path(), is_full_path: bool = False
    ) -> None:
        """
        Save the current module to the specified path.
        Args:
            module_path (Path, optional): The path where the module should be saved. Defaults to an empty Path.
            is_full_path (bool, optional): Flag indicating whether the provided path is a full path. Defaults to False.
        Returns:
            None
        """
        super().save_module(module_path, is_full_path)
        model_file = self.module_path / "model_state.joblib"
        joblib.dump(self.engine_model, model_file)
        super().dump_info_report_assets()

    def load_module(self, module_path: Path, is_full_path: bool = False) -> None:
        """
        Loads a machine learning model from the specified module path.
        Args:
            module_path (Path): The path to the directory containing the model files.
            is_full_path (bool, optional): If True, the provided module_path is considered as the full path to the model file. Defaults to False.
        Raises:
            ValueError: If the provided module_path does not contain a 'model_state.joblib' file.
        """
        super().load_module(module_path, is_full_path)
        model_file = self.module_path / "model_state.joblib"
        # Check that the module_path of the to-be-loaded model contains the necessary files
        if not (model_file).exists():
            raise ValueError(
                f"The provided module_path: '{self.module_path}' doesn't contain a model_state.pkl file"
            )

        self.engine_model = joblib.load(model_file)
