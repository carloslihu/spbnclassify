"""Noisy-OR and Noisy-AND canonical models for binary variables.

They approximate the CPT P(Y | X_1, ..., X_n) of a binary child Y from the a priori
marginals P(X_i = 1) and P(Y = 1) only. With independent parents, the marginals give a
single equation, so the per-parent strengths are parameterized as ``scale * w_i``, where
``w_i`` are relative weights (equal by default) and ``scale`` is solved so that the
model reproduces P(Y = 1) exactly.

Noisy-OR (any active parent can switch Y on, leak p_0):
    P(Y=1 | x) = 1 - (1 - p_0) * prod_{i: x_i=1} (1 - p_i)
    P(Y=1)     = 1 - (1 - p_0) * prod_i (1 - p_i * P(X_i=1))

Noisy-AND (every inactive parent can switch Y off, inhibitor leak l), the dual of the
Noisy-OR on the negated variables:
    P(Y=1 | x) = (1 - l) * prod_{i: x_i=0} (1 - q_i)
    P(Y=1)     = (1 - l) * prod_i (1 - q_i * P(X_i=0))
"""

import itertools

import numpy as np
import pandas as pd
import pyagrum as gum
from scipy.optimize import brentq

PROB_FALSE_COLUMN = "p_false"
PROB_TRUE_COLUMN = "p_true"


class UnreachableTargetError(ValueError):
    """Raised when no link strengths in [0, 1] reproduce the target P(Y = 1)."""

    def __init__(self, target_prior: float, low: float, high: float, hint: str) -> None:
        self.target_prior = target_prior
        self.low = low
        self.high = high
        super().__init__(
            f"P(Y=1)={target_prior:.6g} is outside the reachable range "
            f"[{low:.6g}, {high:.6g}] for the given priors, leak and weights. {hint}"
        )


class _NoisyGate:
    """Common CPT enumeration and pyagrum export. Subclasses define ``parents`` and ``prob_true``."""

    parents: list[str]

    def prob_true(self, x: np.ndarray) -> np.ndarray:
        """Returns P(Y=1 | x) for binary parent configurations x of shape (n_configs, n_parents)."""
        raise NotImplementedError

    def cpt(self) -> pd.DataFrame:
        """Returns the CPT with one row per parent configuration (0 = inactive, 1 = active).

        Returns:
            pd.DataFrame: Parent columns followed by the P(Y=0 | x) and P(Y=1 | x) columns.
        """
        configs = np.array(
            list(itertools.product([0, 1], repeat=len(self.parents))), dtype=int
        ).reshape(-1, len(self.parents))
        p_true = self.prob_true(configs)
        df = pd.DataFrame(configs, columns=self.parents)
        df[PROB_FALSE_COLUMN] = 1 - p_true
        df[PROB_TRUE_COLUMN] = p_true
        return df

    def fill_cpt(
        self,
        bn: gum.BayesNet,
        child: str,
        active_states: dict[str, str] | None = None,
    ) -> None:
        """Fills the CPT of ``child`` in a pyagrum BayesNet whose arcs are already set.

        Args:
            bn (gum.BayesNet): Bayesian network containing the child and its parents.
            child (str): Name of the child variable.
            active_states (dict[str, str] | None, optional): Label of the active (true) state
                for each variable. Variables not listed use their second label (index 1).
                Defaults to None.

        Raises:
            ValueError: If the parents in the network differ from the model parents, or if a
                variable is not binary.
        """
        active_states = active_states or {}
        bn_parents = {bn.variable(i).name() for i in bn.parents(child)}
        if bn_parents != set(self.parents):
            raise ValueError(
                f"Parents of '{child}' in the network {sorted(bn_parents)} "
                f"do not match the model parents {sorted(self.parents)}"
            )

        def active_index(name: str) -> int:
            variable = bn.variable(name)
            if variable.domainSize() != 2:
                raise ValueError(f"Variable '{name}' is not binary")
            if name in active_states:
                return variable.index(active_states[name])
            return 1

        child_active = active_index(child)
        parents_active = {p: active_index(p) for p in self.parents}
        cpt = bn.cpt(child)
        for row in self.cpt().to_dict("records"):
            instantiation = {
                p: parents_active[p] if row[p] == 1 else 1 - parents_active[p]
                for p in self.parents
            }
            probs = [0.0, 0.0]
            probs[child_active] = row[PROB_TRUE_COLUMN]
            probs[1 - child_active] = row[PROB_FALSE_COLUMN]
            cpt[instantiation] = probs


class NoisyOR(_NoisyGate):
    """Noisy-OR CPT fitted to the a priori marginals of the parents and the child.

    Attributes:
        parents (list[str]): Parent names, in CPT column order.
        priors (np.ndarray): P(X_i = 1) for each parent.
        target_prior (float): P(Y = 1).
        leak (float): P(Y = 1) when no parent is active.
        weights (np.ndarray): Relative link strengths, normalized to a maximum of 1.
        scale (float): Solved factor such that links = scale * weights.
        links (np.ndarray): p_i, P(Y = 1) when only X_i is active and there is no leak.
    """

    def __init__(
        self,
        parent_priors: dict[str, float],
        target_prior: float,
        leak: float = 0.01,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Fits the Noisy-OR so that its marginal P(Y = 1) equals ``target_prior``.

        Args:
            parent_priors (dict[str, float]): P(X_i = 1) for each parent.
            target_prior (float): P(Y = 1).
            leak (float, optional): P(Y = 1) when no parent is active. Defaults to 0.01.
            weights (dict[str, float] | None, optional): Relative strength of each parent link.
                Defaults to None (equal strengths).

        Raises:
            ValueError: If the inputs are not probabilities.
            UnreachableTargetError: If no link strengths in [0, 1] reproduce ``target_prior``
                with the given leak and weights.
        """
        self.parents = list(parent_priors)
        self.priors = np.array([parent_priors[p] for p in self.parents], dtype=float)
        self.target_prior = float(target_prior)
        self.leak = float(leak)
        self.weights = self._normalize_weights(weights)
        _check_probabilities(self.priors, "parent priors")
        _check_probabilities(
            np.array([self.target_prior, self.leak]), "target prior and leak"
        )
        if self.leak >= 1:
            raise ValueError("The leak must be lower than 1")
        self.scale = self._solve_scale()
        self.links = self.scale * self.weights

    def _normalize_weights(self, weights: dict[str, float] | None) -> np.ndarray:
        if weights is None:
            return np.ones(len(self.parents))
        w = np.array([weights[p] for p in self.parents], dtype=float)
        if np.any(w < 0) or not np.any(w > 0):
            raise ValueError(
                "Weights must be non-negative with at least one positive value"
            )
        return w / w.max()

    def _marginal(self, scale: float) -> float:
        return float(
            1 - (1 - self.leak) * np.prod(1 - scale * self.weights * self.priors)
        )

    def _solve_scale(self) -> float:
        low, high = self._marginal(0.0), self._marginal(1.0)
        # Tight tolerances: priors of rare events (e.g. 1e-5) must not snap to a bound
        if np.isclose(self.target_prior, low, rtol=1e-9, atol=1e-12):
            return 0.0
        if np.isclose(self.target_prior, high, rtol=1e-9, atol=1e-12):
            return 1.0
        if not low <= self.target_prior <= high:
            raise UnreachableTargetError(
                self.target_prior,
                low,
                high,
                "Lower the leak if P(Y=1) is too small; the parents cannot explain "
                "P(Y=1) if it is too large.",
            )
        return float(brentq(lambda s: self._marginal(s) - self.target_prior, 0.0, 1.0))

    def marginal(self) -> float:
        """Returns P(Y = 1) implied by the fitted model and the parent priors."""
        return self._marginal(self.scale)

    def prob_true(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return 1 - (1 - self.leak) * np.prod(
            np.where(x == 1, 1 - self.links, 1.0), axis=1
        )


class NoisyAND(_NoisyGate):
    """Noisy-AND CPT fitted to the a priori marginals of the parents and the child.

    It is fitted as a Noisy-OR on the negated variables (Y=0 given the inactive parents).

    Attributes:
        parents (list[str]): Parent names, in CPT column order.
        priors (np.ndarray): P(X_i = 1) for each parent.
        target_prior (float): P(Y = 1).
        inhibitor (float): P(Y = 0) when all parents are active.
        weights (np.ndarray): Relative inhibition strengths, normalized to a maximum of 1.
        scale (float): Solved factor such that inhibitions = scale * weights.
        inhibitions (np.ndarray): q_i, P(Y = 0) when only X_i is inactive and there is no
            inhibitor leak.
    """

    def __init__(
        self,
        parent_priors: dict[str, float],
        target_prior: float,
        inhibitor: float = 0.01,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Fits the Noisy-AND so that its marginal P(Y = 1) equals ``target_prior``.

        Args:
            parent_priors (dict[str, float]): P(X_i = 1) for each parent.
            target_prior (float): P(Y = 1).
            inhibitor (float, optional): P(Y = 0) when all parents are active. Defaults to 0.01.
            weights (dict[str, float] | None, optional): Relative strength with which each
                inactive parent switches Y off. Defaults to None (equal strengths).

        Raises:
            ValueError: If the inputs are not probabilities.
            UnreachableTargetError: If no inhibition strengths in [0, 1] reproduce
                ``target_prior`` with the given inhibitor and weights.
        """
        _check_probabilities(
            np.array(list(parent_priors.values()), dtype=float), "parent priors"
        )
        _check_probabilities(np.array([target_prior]), "target prior")
        try:
            self._dual = NoisyOR(
                parent_priors={p: 1 - prior for p, prior in parent_priors.items()},
                target_prior=1 - target_prior,
                leak=inhibitor,
                weights=weights,
            )
        except UnreachableTargetError as e:
            raise UnreachableTargetError(
                target_prior,
                1 - e.high,
                1 - e.low,
                "Lower the inhibitor if P(Y=1) is too large; the parents cannot explain "
                "P(Y=1) if it is too small.",
            ) from None
        self.parents = self._dual.parents
        self.priors = 1 - self._dual.priors
        self.target_prior = float(target_prior)
        self.inhibitor = self._dual.leak
        self.weights = self._dual.weights
        self.scale = self._dual.scale
        self.inhibitions = self._dual.links

    def marginal(self) -> float:
        """Returns P(Y = 1) implied by the fitted model and the parent priors."""
        return 1 - self._dual.marginal()

    def prob_true(self, x: np.ndarray) -> np.ndarray:
        return 1 - self._dual.prob_true(1 - np.atleast_2d(x))


class NoisyProduct(_NoisyGate):
    """Conjunction of independent noisy gates over disjoint parents.

    Y = 1 only if every gate outputs 1, so P(Y=1 | x) = prod_g P_g(Y=1 | x_g) and, with independent parents, P(Y=1) = prod_g P_g(Y=1).

    Attributes:
        gates (list[NoisyOR | NoisyAND]): The combined gates.
        parents (list[str]): Parents of all gates, in gate order.
    """

    def __init__(self, gates: list["NoisyOR | NoisyAND"]) -> None:
        """Combines the gates.

        Args:
            gates (list[NoisyOR | NoisyAND]): Fitted gates with disjoint parents.

        Raises:
            ValueError: If the gates share parents.
        """
        self.gates = list(gates)
        self.parents = [p for gate in self.gates for p in gate.parents]
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("The gates must have disjoint parents")

    def marginal(self) -> float:
        """Returns P(Y = 1) implied by the fitted gates and the parent priors."""
        return float(np.prod([gate.marginal() for gate in self.gates]))

    def prob_true(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        result = np.ones(len(x))
        start = 0
        for gate in self.gates:
            end = start + len(gate.parents)
            result *= gate.prob_true(x[:, start:end])
            start = end
        return result


def _check_probabilities(values: np.ndarray, name: str) -> None:
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"The {name} must be probabilities in [0, 1]")
