import pandas as pd
import pybnesian as pbn

from .base import BayesianNetwork
from .semiparametric import SemiParametricBayesianNetwork


class KDEBayesianNetwork(SemiParametricBayesianNetwork):
    # bn_type = pbn.KDENetworkType() # NOTE: To allow hybrid KDE
    search_operators = ["arcs"]

    def __str__(self) -> str:
        """Returns the string representation of the SemiParametric Bayesian Network

        Returns:
            str: The string representation
        """
        return "KDE " + BayesianNetwork.__str__(self)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pbn.BayesianNetwork:
        nodes = X.columns
        self.type_whitelist = [
            (node, pbn.CKDEType()) for node in nodes if node != self.true_label
        ]
        super().fit(X, y)
        return self
