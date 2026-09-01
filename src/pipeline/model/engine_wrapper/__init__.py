from .bayesian_network import BNCModel
from .sklearn import SKLEARN_MODELS, SklearnModel

classifier_wrapper_dict = {
    "BNC": BNCModel,
    **{key: SklearnModel for key in SKLEARN_MODELS},  # Add all sklearn models
}
