from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .module import Module


class CompositeModule(Module):
    """Modular class to build a composite architecture of modules from a config file, which contains the arguments for each module.
    The modules are stored in a dictionary, where the key is the module name and the value is the module object.

    Attributes:
        modules_dict (dict): Defaultdict that will incorporate the modules that can be built into the composite structure and their key.
        modules (dict): Dictionary with the modules of the composite architecture. The keys are the module names and the values are the module objects.
        config_dict (dict, inherited): Dictionary with the configuration of the composite architecture.

    Private Methods:
        __build_modules: Builds and initializes the modules of the composite architecture.
        __can_execute: Checks if a given method can be executed on a module.

    Protected Methods:
        _get_module_tree: Returns the tree of modules of the composite architecture as a string.
        _broadcast_execute: Broadcasts the execution of a method on all the nested modules of the composite architecture over the same inputs.
        _series_execute: Broadcasts the execution of a method on all the nested modules of the composite architecture over the same inputs, in series.
        _join_metrics: Joins the metrics of all the modules in the composite architecture.

    Public Methods:
        save_module: Creates a directory for saving a module and dumps the Module config dictionary in a json file.
        load_module: Loads the module from the base path where the submodules are saved.

    """

    # Building modules dictionary: contains the modules that can be built into the composite structure and their key
    # It is a defaultdict with a lambda function that returns None as default value
    modules_dict = defaultdict(lambda: None)

    def __init__(self, config_dict: dict, module_id: str, level: int) -> None:
        """Initializes the CompositeModule class.

        Args:
            config_dict (dict): Dictionary with the configuration of the architecture.
            The keys are the module names and the values are the configuration dictionaries for each module.
            module_id (str): Identifier of the composite module.
            level (int): Level of the composite module in the architecture.
        """
        # Inherited initialization: stores the configuration dictionary as an attribute
        super().__init__(config_dict=config_dict, module_id=module_id, level=level)
        # Build the architecture of modules
        self.modules = self.__build_modules(config_dict)

    def save_module(
        self,
        module_path: Path = Path(),
        is_full_path: bool = False,
    ) -> None:
        """Saves the composite module in the specified path and stores the directory path as an attribute.
        This class extends the save_module method of the parent Module class: it saves the composite module
        as a module and then executes the save method on all of the nested modules.

        Args:
            module_path (str): Base path to save the composite module at.
        """
        # Call the save method of the Module class to save the composite module
        super().save_module(module_path, is_full_path)
        # Distributedly execute the save method on all the nested modules
        _ = self._broadcast_execute("save_module", module_path=self.module_path)
        # Submodules are now saved in the directory of the parent composite module!

    def load_module(self, module_path: Path, is_full_path: bool = False) -> None:
        """Loads the submodule structure of the composite module from the base path where the submodules are saved.
        This class extends the load_module method of the Module class, so it also stores the module path as an attribute.

        Args:
            module_path (str): Base path where the composite module is saved.
            is_full_path (bool, optional): Flag to indicate if the full path is provided. Defaults to False.
            This flag is used to indicate if the module ID should be appended to the module path, and it is needed for
            every submodule in a modular structure stemming from a top-level composite module. In practice it is used
            for every module and composite module that is not the top-level pipeline.

        """
        super().load_module(module_path, is_full_path)
        # Distributedly execute the load method on all the nested modules
        _ = self._broadcast_execute("load_module", module_path=self.module_path)
        # Submodules are now initialized!

    def _get_module_tree(self) -> str:
        """Returns the tree of modules of the a CompositeModule object as a string.
        Each tree node contains the module level, its ID and the class name.
        Format: "L{level}: {module_id} ({module_class_name})".

        Returns:
            str: Tree-like string with the modules of the pipeline.
        """

        def __get_submodule_tree(module: CompositeModule, base_level: int) -> str:
            """Returns the tree of submodules of a given module as a string. It can be called recursively to get the full tree.
            The tree is returned relative to the level of reference, defined by the base_level argument.

            Args:
                module (CompositeModule): CompositeModule object to get the tree from.
                base_level (int): Level of reference for the tree.

            Returns:
                str: _description_
            """
            # Get the items of the submodules dictionary
            submodules_dict_items = module.modules.items()
            # Initialize an empty string to store the subtree
            subtree_str = ""
            # Iterate over the submodules and add them to the subtree string
            for idx, (submodule_id, submodule) in enumerate(submodules_dict_items):
                # Calculate the relative level of the submodule
                relative_level = submodule.level - base_level
                # Get the class name of the submodule and convert it to snake case
                module_name = self.camel_to_snake(type(submodule).__name__)
                # Define the connector for the tree node (├── generally, and └── if it is the last submodule of its level)
                connector = "├── " if idx < len(submodules_dict_items) - 1 else "└── "
                # Add spacing
                subtree_str += "│   " * (relative_level) + "\n"
                # Add the tree node to the subtree string
                subtree_str += (
                    "│   " * (relative_level - 1)
                    + connector
                    + f"L{submodule.level}: {submodule_id} ({module_name})"
                    + "\n"
                )
                # If the submodule is a CompositeModule, call the function recursively
                if isinstance(submodule, CompositeModule):
                    subtree_str += __get_submodule_tree(submodule, base_level)
            # Return the subtree string
            return subtree_str

        # Get the class name of the pipeline and convert it to snake case
        class_name = self.camel_to_snake(type(self).__name__)
        # Get the base level of the tree (the level of the calling module)
        base_level = self.level
        # Initialize the tree string with the root module
        tree_str = f"L{base_level}: {self.module_id} ({class_name})\n"
        # Add the tree of submodules to the tree string
        tree_str += __get_submodule_tree(self, base_level)
        # Remove trailing \n
        tree_str = tree_str.rstrip()
        # Return the tree of modules as a string
        return tree_str

    def _broadcast_execute(
        self, method_name: str, method_absent_strategy: str = "", **kwargs
    ) -> dict:
        """Broadcast the execution of a method (if it exists) on all the nested modules of the composite architecture over the same inputs .

        The selected method is executed on all the modules of the composite architecture in parallel with respect to the data/argument workflow (do not
        confuse with parallel processing in a computational sense), i.e., the method is executed on the same input data/arguments for all the modules,
        and the results are collected in a flat (single-level) dictionary.

        Methods are only distributed to the submodules in the immediately below level of the composite module that is calling the method.
        Note that this method is safe to use with methods that are not implemented in all the modules, as it can raise,
        warn, or ignore the corresponding errors.

        Args:
            method_name (str): Name of the method to be executed.
            method_absent_strategy (str, optional): How to handle absent method strategy. Defaults to "" (ignore and continue). Can also be set to throw a warning or an error, or to "propagate" (send the method to the next level).

        Returns:
            dict: Dictionary with the results of the method execution on all of the modules.
            The keys are the submodule names and the values are the returned results of the method execution on each submodule.
        """

        # Create empty result dictionary
        result = {}
        # Iterate over the modules and execute the method if it exists
        for module_id, module in self.modules.items():
            # Check if the method can be executed on the module
            if self.__can_execute(module, method_name):
                # If able, execute the method with the given kwargs
                result[module_id] = getattr(module, method_name)(**kwargs)
            elif method_absent_strategy == "raise":
                # If not able, raise an error
                raise AttributeError(
                    f"Method {method_name} is not implemented in {type(module).__name__}."
                )
            elif method_absent_strategy == "warn":
                # If not able, raise a warning
                self.log_note(
                    self.level,
                    f"Method {method_name} is not implemented in {type(module).__name__}.",
                    category="WARNING",
                )
            elif method_absent_strategy == "propagate":
                if isinstance(module, CompositeModule):
                    result[module_id] = module._broadcast_execute(
                        method_name, method_absent_strategy, **kwargs
                    )
        # Return the result dictionary
        return result

    def _series_execute(
        self, method_name: str, alternate_method: str = "", **kwargs
    ) -> pd.DataFrame:
        """Broadcast the execution of a method (if it exists) on all the nested modules of the composite architecture over the same inputs .

        The selected method is executed on all the modules of the composite architecture in parallel with respect to the data/argument workflow (do not
        confuse with parallel processing in a computational sense), i.e., the method is executed on the same input data/arguments for all the modules,
        and the results are collected in a flat (single-level) dictionary.

        An alternate method can be specified to be executed if the main method is not implemented in any of the modules.

        Methods are only distributed to the submodules in the immediately below level of the composite module that is calling the method.
        Note that this method is safe to use with methods that are not implemented in all the modules, as it can raise,
        warn, or ignore the corresponding errors.

        It raises an error if the method is not implemented in any of the modules.

        Args:
            method_name (str): Name of the method to be executed.
            alternate_method (str): Name of the alternate method to be executed if the main method is not implemented in any of the modules.
            it defaults to an empty string, which means that an error will be raised if the main method is not implemented for some module.
            error (str, optional): Error handling strategy. Defaults to "" (ignore).

        Returns:
            pd.DataFrame: Dataframe with the results of the method execution on all of the modules.
            The keys are the submodule names and the values are the returned results of the method execution on each submodule.
        """
        # NOTE: Assumes that modules are ordered for series execution
        # Iterate over the modules and execute the method if it exists
        for module_id, module in self.modules.items():
            # Check if the main method can be executed on the module
            if self.__can_execute(module, method_name):
                # If able, execute the method with the given kwargs and update "data" in kwargs
                # RFE: que "data" no sea un argumento fijo, sino que se pueda elegir qué key de kwargs se actualiza
                kwargs["data"] = getattr(module, method_name)(**kwargs)
            else:
                if alternate_method:
                    if self.__can_execute(module, alternate_method):
                        kwargs["data"] = getattr(module, alternate_method)(**kwargs)
                    else:
                        # If not able to execute, raise an error
                        raise AttributeError(
                            f"Method {method_name} and alternate method {alternate_method} are not implemented in {type(module).__name__}."
                        )
                else:
                    # If not able to execute, raise an error
                    raise AttributeError(
                        f"Method {method_name} is not implemented in {type(module).__name__}."
                    )
        # Return the result dictionary
        # TODO: usar los metadatos de los objectos para pasar info de ejecución
        return kwargs["data"]

    def _join_metrics(self) -> dict[str, Any]:
        """
        Aggregates and combines metrics from the current module and its superclasses.
        This method executes the '_join_metrics' method across all broadcasted modules,
        merges the resulting metric dictionaries, and optionally overwrites and dumps
        the combined metrics to a JSON file.
        Returns:
            dict[str, Any]: The combined dictionary of metrics from all relevant modules.
        """

        combined_metric_dict = self._broadcast_execute("_join_metrics")
        combined_metric_dict.update(super()._join_metrics())
        self._dump_object(
            combined_metric_dict,
            self.module_path,
            "metrics",
            Path("output"),
            force_format="json",
        )

        return combined_metric_dict

    def __build_modules(self, config_dict: dict) -> dict:
        """Builds and initializes the modules of the composite architecture.

        Args:
            config_dict (dict): Dictionary with the configuration of the composite architecture.

        Returns:
            dict: Dictionary containing the modules of the composite architecture.
            The keys are the module names and the values are the module objects.
        """

        def is_valid_module_config(module_config):
            """Check if the module configuration is valid.
            If the module_config is false, null or not in the config, it will be skipped.
                It does not raise a warning because it could be a placeholder for a submodule

            If it is not a dictionary, it will be skipped too (can't be a submodule).
                It does not raise a warning because it could be an attribute of the parent module
            """
            return isinstance(module_config, dict) and "module_type" in module_config

        def is_callable_module(module_type):
            """Check if the module is in the modules_dict of the CompositeModule and
            if the module_type is callable."""
            return module_type in self.modules_dict and callable(
                self.modules_dict[module_type]
            )

        # Initialize an empty dictionary to store the modules
        modules = {}

        for module_id, module_config in config_dict.items():
            if not is_valid_module_config(module_config):
                # This check does not raise a warning because it could be a placeholder for a submodule,
                # or an attribute of the parent module
                continue

            # As the module configuration is valid, extract the module type from the configuration dictionary
            module_type = module_config["module_type"]

            if not is_callable_module(module_type):
                # This check raises a warning because it is a misconfiguration of the pipeline
                self.log_note(
                    self.level,
                    f"Module {module_type} is not callable or not part of the modules_dict of {type(self).__name__}, skipping.",
                    category="ERROR",
                )
                continue

            # If the module is part of the modules_dict of the CompositeModule, it will be built from the class
            # specified in the modules_dict dictionary and the configuration dictionary extracted from the config_dict
            module_loader = self.modules_dict.get(module_type)
            # If the module_loader is None, skip the module
            if module_loader is None:
                # This check raises a warning because it is a misconfiguration of the pipeline
                self.log_note(
                    self.level,
                    f"Module loader for {module_type} is None, skipping.",
                    category="WARNING",
                )
                continue

            # If the module is preloaded, the config_dict must be loaded from the specified path
            if "preloaded" in module_config and module_config["preloaded"]:
                preloaded_path = Path(module_config["preloaded"])
                module_loader_name = self.camel_to_snake(module_loader.__name__)
                module_config = self._load_object(
                    f"{module_loader_name}_config", "json", preloaded_path
                )
                module_config["preloaded"] = True
            else:
                preloaded_path = None

            # Build from the class specified in the modules_dict dictionary and the configuration dictionary
            modules[module_id] = module_loader(module_config, module_id, self.level + 1)

            # If the module is preloaded, it must be loaded from the specified path
            # TODO: que no entrene ni se ajuste ni nada
            # TODO: que se guarde un flag de que es preloaded
            # TODO: que se guarde la info del módulo precargado en todos los niveles superiores
            # TODO: que si los módulos tienen las scores, no las recalculen?...
            if "preloaded" in module_config and module_config["preloaded"]:
                modules[module_id].load_module(preloaded_path, is_full_path=True)

        # Return the dictionary with the retrieved modules
        return modules

    def __can_execute(self, module: object, method_name: str) -> bool:
        """Checks if a given method can be executed on a module.

        Args:
            module (object): Module object.
            method_name (str): Name of the method to be executed.

        Returns:
            bool: True if the method can be executed on the module, False otherwise.
        """
        # Check if the method exists in the module
        return callable(getattr(module, method_name, None))
