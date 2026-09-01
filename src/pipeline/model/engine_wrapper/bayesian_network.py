from pathlib import Path

from ....bnc import BNC_MODEL_CLASS_DICT
from ..single_model import ClassifierSingleModel


class BNCModel(ClassifierSingleModel):
    """
    BNCModel is a wrapper class for Bayesian Network Classifier (BNC) models within a modular pipeline framework.
    It extends the ClassifierSingleModel base class and provides custom initialization, saving, and loading
    functionality specific to Bayesian Network models.
    Attributes:
        engine_model_class: The class of the underlying engine model, determined by the 'parametric' and 'structure'
            keys in the configuration dictionary.
        engine_model: The instantiated engine model object.
    Methods:
        __init__(config_dict: dict, module_id: str, level: int) -> None
            Initializes the BNCModel with the provided configuration, ensuring required model arguments are present.
        save_module(module_path: Path = Path(), is_full_path: bool = False) -> None
            Saves the current state of the module, including the engine model and associated assets, to disk.
        load_module(module_path: Path, is_full_path: bool = False) -> None
            Loads the module and its engine model state from the specified path, restoring the model for use.
    """

    def __init__(self, config_dict: dict, module_id: str, level: int) -> None:
        """
        Initializes the Bayesian Network engine wrapper with the specified configuration.
        Args:
            config_dict (dict): Configuration dictionary containing model arguments. Must include
                'structure' and 'parametric' keys within the 'model_args' sub-dictionary.
            module_id (str): Identifier for the module instance.
            level (int): The level or depth of the module in the pipeline.
        Raises:
            ValueError: If 'structure' or 'parametric' keys are missing from config_dict['model_args'].
        """

        structure = config_dict["model_args"].get("structure", None)
        if structure is None:
            raise ValueError(
                f"The config_dict for a {self.__str__()} must contain the key 'structure' in its model_args dict"
            )

        parametric = config_dict["model_args"].get("parametric", None)
        if parametric is None:
            raise ValueError(
                f"The config_dict for a {self.__str__()} must contain the key 'parametric' in its model_args dict"
            )

        self.engine_model_class = BNC_MODEL_CLASS_DICT[parametric + structure]
        super().__init__(config_dict, module_id, level)

    def save_module(
        self, module_path: Path = Path(), is_full_path: bool = False
    ) -> None:
        """
        Saves the current state of the module, including the engine model and associated assets.
        Args:
            module_path (Path, optional): The directory or file path where the module should be saved. Defaults to the current directory.
            is_full_path (bool, optional): Indicates whether `module_path` is a full file path or a directory. Defaults to False.
        Returns:
            None
        Side Effects:
            - Calls the parent class's `save_module` method.
            - Saves the engine model's state to a pickle file named 'model_state.pkl' within the module path.
            - Dumps additional information and report assets using the parent class's `dump_info_report_assets` method.
        """
        super().save_module(module_path, is_full_path)
        model_file = self.module_path / "model_state.pkl"
        self.engine_model.save(model_file)
        super().dump_info_report_assets()

    def load_module(self, module_path: Path, is_full_path: bool = False) -> None:
        """
        Loads a model module from the specified path.
        This method first calls the superclass's `load_module` method to perform any necessary setup.
        It then checks for the existence of a 'model_state.pkl' file within the module path. If the file
        does not exist, a ValueError is raised. If the file exists, the model state is loaded using
        the engine model class.
        Args:
            module_path (Path): The path to the module directory or file.
            is_full_path (bool, optional): Whether the provided path is a full path to the module. Defaults to False.
        Raises:
            ValueError: If the 'model_state.pkl' file is not found in the specified module path.
        """

        super().load_module(module_path, is_full_path)
        model_file = self.module_path / "model_state.pkl"
        # Check that the module_path of the to-be-loaded model contains the necessary files
        if not (model_file).exists():
            raise ValueError(
                f"The provided module_path: '{self.module_path}' doesn't contain a model_state.pkl file"
            )

        self.engine_model = self.engine_model_class.load(model_file)
