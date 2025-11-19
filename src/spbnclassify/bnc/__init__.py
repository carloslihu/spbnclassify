import inspect

from .aode import (
    GaussianAveragedOneDependenceEstimator,
    KDEAveragedOneDependenceEstimator,
    SemiParametricAveragedOneDependenceEstimator,
)
from .banc import (
    GaussianBayesianNetworkAugmentedNaiveBayes,
    KDEBayesianNetworkAugmentedNaiveBayes,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
)
from .base import BaseBayesianNetworkClassifier
from .bmc import (
    GaussianBayesianMultinet,
    KDEBayesianMultinet,
    SemiParametricBayesianMultinet,
)
from .kdbnc import (
    GaussianKDependenceBayesian,
    KDEKDependenceBayesian,
    SemiParametricKDependenceBayesian,
)
from .maxkbnc import (
    GaussianMaxKAugmentedNaiveBayes,
    KDEMaxKAugmentedNaiveBayes,
    SemiParametricMaxKAugmentedNaiveBayes,
)
from .nbnc import GaussianNaiveBayes, KDENaiveBayes, SemiParametricNaiveBayes
from .snbnc import (
    GaussianSelectiveNaiveBayes,
    KDESelectiveNaiveBayes,
    SemiParametricSelectiveNaiveBayes,
)
from .spodebnc import (
    GaussianSuperParentOneDependenceEstimator,
    KDESuperParentOneDependenceEstimator,
    SemiParametricSuperParentOneDependenceEstimator,
)
from .tanbnc import (
    GaussianTreeAugmentedNaiveBayes,
    KDETreeAugmentedNaiveBayes,
    SemiParametricTreeAugmentedNaiveBayes,
)

# from .discrete_bayesian_network_classifier import (
#     DiscreteBayesianMultinet,
#     DiscreteBayesianNetworkClassifier,
#     DiscreteMaxKAugmentedNaiveBayes,
#     DiscreteNaiveBayes,
#     DiscreteSelectiveNaiveBayes,
#     DiscreteSuperParentOneDependenceEstimator,
#     DiscreteTreeAugmentedNaiveBayes,
# )


# Automatically build BNC_MODEL_CLASS_DICT from imported classifier classes
BNC_MODEL_CLASS_DICT = {
    name: obj
    for name, obj in globals().items()
    if inspect.isclass(obj) and name != "BaseBayesianNetworkClassifier"
}

__all__ = [
    "BaseBayesianNetworkClassifier",
    # Gaussian classifiers
    "GaussianAveragedOneDependenceEstimator",
    "GaussianBayesianMultinet",
    "GaussianBayesianNetworkAugmentedNaiveBayes",
    "GaussianKDependenceBayesian",
    "GaussianMaxKAugmentedNaiveBayes",
    "GaussianNaiveBayes",
    "GaussianSelectiveNaiveBayes",
    "GaussianSuperParentOneDependenceEstimator",
    "GaussianTreeAugmentedNaiveBayes",
    # KDE classifiers
    "KDEAveragedOneDependenceEstimator",
    "KDEBayesianMultinet",
    "KDEBayesianNetworkAugmentedNaiveBayes",
    "KDEKDependenceBayesian",
    "KDEMaxKAugmentedNaiveBayes",
    "KDENaiveBayes",
    "KDESelectiveNaiveBayes",
    "KDESuperParentOneDependenceEstimator",
    "KDETreeAugmentedNaiveBayes",
    # SemiParametric classifiers
    "SemiParametricAveragedOneDependenceEstimator",
    "SemiParametricBayesianMultinet",
    "SemiParametricBayesianNetworkAugmentedNaiveBayes",
    "SemiParametricKDependenceBayesian",
    "SemiParametricMaxKAugmentedNaiveBayes",
    "SemiParametricNaiveBayes",
    "SemiParametricSelectiveNaiveBayes",
    "SemiParametricSuperParentOneDependenceEstimator",
    "SemiParametricTreeAugmentedNaiveBayes",
    # Model class dictionary for engine wrapper
    "BNC_MODEL_CLASS_DICT",
]
