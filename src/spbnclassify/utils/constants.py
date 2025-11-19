import numpy as np
import pybnesian as pbn

N_NEIGHBORS = 3

CONTINUOUS_NODES = ["a", "b", "c", "d", "e"]
PROB_GAUSSIAN = 0.5  # Probability of generating a parametric node
PROB_DISCRETE_DISCRETE = (
    0.5  # Probability of generating an arc between two discrete nodes
)
PROB_DISCRETE_CONTINUOUS = (
    0.75  # Probability of generating an arc between a discrete and a continuous node
)
PROB_CONTINUOUS_CONTINUOUS = (
    0.5  # Probability of generating an arc between two continuous nodes
)

NAN_LOGL_VALUE = np.log(1e-10)  # Used to replace NaN log-likelihood values

# TODO: Add legend in plots
NODE_TYPE_COLOR_MAP = {
    pbn.DiscreteFactorType(): 0.1,  # RED
    pbn.LinearGaussianCPDType(): 0.5,  # YELLOW
    pbn.CKDEType(): 0.2,  # BLUE
}
