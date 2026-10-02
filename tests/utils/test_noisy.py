import numpy as np
import pyagrum as gum
import pytest
from src.utils.noisy import (
    PROB_FALSE_COLUMN,
    PROB_TRUE_COLUMN,
    NoisyAND,
    NoisyOR,
    NoisyProduct,
    UnreachableTargetError,
)

PARENT_PRIORS = {"a": 0.1, "b": 0.3, "c": 0.05}
TARGET_PRIOR = 0.2
LEAK = 0.02
INHIBITOR = 0.05


def cpt_marginal(model: NoisyOR | NoisyAND) -> float:
    """Computes P(Y=1) by summing the CPT over independent parents."""
    cpt = model.cpt()
    priors = np.array([PARENT_PRIORS[p] for p in model.parents])
    configs = cpt[model.parents].to_numpy()
    weights = np.prod(np.where(configs == 1, priors, 1 - priors), axis=1)
    return float(weights @ cpt[PROB_TRUE_COLUMN].to_numpy())


def build_bn(child_labels: list[str] | None = None) -> gum.BayesNet:
    bn = gum.BayesNet()
    for name, prior in PARENT_PRIORS.items():
        bn.add(gum.LabelizedVariable(name, name, 2))
        bn.cpt(name).fillWith([1 - prior, prior])
    child = gum.LabelizedVariable("y", "y", 0)
    for label in child_labels or ["0", "1"]:
        child.addLabel(label)
    bn.add(child)
    for name in PARENT_PRIORS:
        bn.addArc(name, "y")
    return bn


def inferred_prior(bn: gum.BayesNet, label: str) -> float:
    ie = gum.LazyPropagation(bn)
    ie.makeInference()
    return ie.posterior("y")[{"y": label}]


@pytest.mark.parametrize("model_class", [NoisyOR, NoisyAND])
class TestNoisyGates:
    def test_marginal_matches_target(self, model_class) -> None:
        model = model_class(PARENT_PRIORS, TARGET_PRIOR)
        assert model.marginal() == pytest.approx(TARGET_PRIOR)
        assert cpt_marginal(model) == pytest.approx(TARGET_PRIOR)

    def test_cpt_is_valid(self, model_class) -> None:
        cpt = model_class(PARENT_PRIORS, TARGET_PRIOR).cpt()
        assert len(cpt) == 2 ** len(PARENT_PRIORS)
        assert np.allclose(cpt[PROB_FALSE_COLUMN] + cpt[PROB_TRUE_COLUMN], 1)
        assert cpt[PROB_TRUE_COLUMN].between(0, 1).all()

    def test_weights_are_respected(self, model_class) -> None:
        weights = {"a": 2.0, "b": 4.0, "c": 1.0}
        model = model_class(PARENT_PRIORS, TARGET_PRIOR, 0.01, weights)
        strengths = model.links if model_class is NoisyOR else model.inhibitions
        assert strengths / strengths.max() == pytest.approx([0.5, 1.0, 0.25])
        assert model.marginal() == pytest.approx(TARGET_PRIOR)

    def test_unreachable_target_raises(self, model_class) -> None:
        with pytest.raises(UnreachableTargetError, match="reachable range"):
            model_class(PARENT_PRIORS, 0.999 if model_class is NoisyOR else 0.001)

    def test_invalid_probabilities_raise(self, model_class) -> None:
        with pytest.raises(ValueError, match="probabilities"):
            model_class({"a": 1.5}, TARGET_PRIOR)

    def test_fill_cpt_reproduces_target(self, model_class) -> None:
        bn = build_bn()
        model_class(PARENT_PRIORS, TARGET_PRIOR).fill_cpt(bn, "y")
        assert inferred_prior(bn, "1") == pytest.approx(TARGET_PRIOR)

    def test_fill_cpt_with_active_states(self, model_class) -> None:
        bn = build_bn(["yes", "no"])
        model = model_class(PARENT_PRIORS, TARGET_PRIOR)
        model.fill_cpt(bn, "y", active_states={"y": "yes"})
        assert inferred_prior(bn, "yes") == pytest.approx(TARGET_PRIOR)
        cpt = bn.cpt("y")
        assert cpt[{"a": 0, "b": 0, "c": 0, "y": "yes"}] == pytest.approx(
            model.prob_true(np.array([0, 0, 0]))[0]
        )

    def test_fill_cpt_parent_mismatch_raises(self, model_class) -> None:
        bn = build_bn()
        bn.eraseArc("c", "y")
        with pytest.raises(ValueError, match="do not match"):
            model_class(PARENT_PRIORS, TARGET_PRIOR).fill_cpt(bn, "y")


class TestNoisyOR:
    def test_leak_is_probability_without_active_parents(self) -> None:
        model = NoisyOR(PARENT_PRIORS, TARGET_PRIOR, leak=LEAK)
        assert model.prob_true(np.zeros(3))[0] == pytest.approx(LEAK)

    def test_single_active_parent(self) -> None:
        model = NoisyOR(PARENT_PRIORS, TARGET_PRIOR, leak=LEAK)
        expected = 1 - (1 - LEAK) * (1 - model.links[1])
        assert model.prob_true(np.array([0, 1, 0]))[0] == pytest.approx(expected)


class TestNoisyAND:
    def test_inhibitor_is_failure_with_all_parents_active(self) -> None:
        model = NoisyAND(PARENT_PRIORS, 0.01, inhibitor=INHIBITOR)
        assert model.prob_true(np.ones(3))[0] == pytest.approx(1 - INHIBITOR)

    def test_single_inactive_parent(self) -> None:
        model = NoisyAND(PARENT_PRIORS, 0.01, inhibitor=INHIBITOR)
        expected = (1 - INHIBITOR) * (1 - model.inhibitions[0])
        assert model.prob_true(np.array([0, 1, 1]))[0] == pytest.approx(expected)

    def test_unreachable_range_is_in_child_terms(self) -> None:
        with pytest.raises(UnreachableTargetError) as e:
            NoisyAND({"a": 0.9, "b": 0.7}, 0.5, inhibitor=INHIBITOR)
        assert e.value.low == pytest.approx((1 - INHIBITOR) * 0.9 * 0.7)
        assert e.value.high == pytest.approx(1 - INHIBITOR)


class TestNoisyProduct:
    OR_PRIORS = {"a": 0.9, "b": 0.8}
    AND_PRIORS = {"c": 0.95}

    def product(self) -> NoisyProduct:
        return NoisyProduct(
            [
                NoisyOR(self.OR_PRIORS, 0.97, leak=LEAK),
                NoisyAND(self.AND_PRIORS, 0.92, inhibitor=INHIBITOR),
            ]
        )

    def test_marginal_is_product_of_gates(self) -> None:
        model = self.product()
        assert model.parents == ["a", "b", "c"]
        assert model.marginal() == pytest.approx(0.97 * 0.92)

    def test_cpt_marginal_matches(self) -> None:
        model = self.product()
        cpt = model.cpt()
        priors = np.array([0.9, 0.8, 0.95])
        configs = cpt[model.parents].to_numpy()
        weights = np.prod(np.where(configs == 1, priors, 1 - priors), axis=1)
        assert weights @ cpt[PROB_TRUE_COLUMN].to_numpy() == pytest.approx(0.97 * 0.92)

    def test_prob_true_multiplies_gates(self) -> None:
        model = self.product()
        or_gate, and_gate = model.gates
        x = np.array([[1, 0, 0]])
        expected = or_gate.prob_true(x[:, :2]) * and_gate.prob_true(x[:, 2:])
        assert model.prob_true(x) == pytest.approx(expected)

    def test_shared_parents_raise(self) -> None:
        with pytest.raises(ValueError, match="disjoint"):
            NoisyProduct(
                [NoisyOR({"a": 0.5}, 0.4), NoisyAND({"a": 0.5}, 0.6)]
            )
