# from .cusum import CUSUMThresholder
from .max_probability import MaxProbabilityThresholder
from .standard import StandardThresholder

thresholder_dict = {
    "standard_thresholder": StandardThresholder,
    "max_probability_thresholder": MaxProbabilityThresholder,
}

__all__ = ["thresholder_dict"]
