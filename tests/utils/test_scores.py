import pandas as pd
import pybnesian as pbn
import pytest
from helpers.data import (
    DATA_SIZE,
    SEED,
    TRUE_CLASS_LABEL,
    generate_normal_data_classification,
)
from src.bnc import GaussianNaiveBayes
from src.utils.scores import (
    AccuracyScore,
    AUCScore,
    ConditionalLogLikelihoodValidatedScore,
    F1Score,
    OracleValidatedScore,
)


class TestOracleValidatedScore:
    @pytest.fixture
    def score(self) -> OracleValidatedScore:
        return OracleValidatedScore()

    @pytest.fixture
    def model(self) -> pbn.GaussianNetwork:
        return pbn.GaussianNetwork(["a", "b", "c", "d"])

    def test_has_variables(self, score: OracleValidatedScore) -> None:
        assert score.has_variables(["a", "d"])
        assert not score.has_variables(["a", "x"])

    def test_compatible_bn(self, score: OracleValidatedScore) -> None:
        assert score.compatible_bn(pbn.GaussianNetwork(["a", "b", "c", "d"]))
        assert not score.compatible_bn(pbn.GaussianNetwork(["a", "b", "c", "x"]))

    @pytest.mark.parametrize(
        "variable,evidence,expected",
        [
            ("c", [], -1.0),
            ("c", ["a"], 0.0),
            ("c", ["b"], 0.5),
            ("c", ["a", "b"], 1.5),
            ("d", ["c"], 1.0),
            ("d", [], -1.0),
            ("a", [], -1.0),
        ],
    )
    def test_local_score_values(
        self,
        score: OracleValidatedScore,
        model: pbn.GaussianNetwork,
        variable: str,
        evidence: list[str],
        expected: float,
    ) -> None:
        assert score.local_score(model, variable, evidence) == expected

    def test_vlocal_score_matches_local_score(
        self, score: OracleValidatedScore, model: pbn.GaussianNetwork
    ) -> None:
        evidence = ["a", "b"]
        assert score.vlocal_score(model, "c", evidence) == score.local_score(
            model, "c", evidence
        )

    def test_structure_learning_prefers_expected_oracle_dag(
        self, score: OracleValidatedScore
    ) -> None:
        hc = pbn.GreedyHillClimbing()
        start_model = pbn.GaussianNetwork(["a", "b", "c", "d"])

        learned_model = hc.estimate(
            operators=pbn.ArcOperatorSet(),
            score=score,
            start=start_model,
            verbose=False,
        )

        assert set(learned_model.arcs()) == {("a", "c"), ("b", "c"), ("c", "d")}


class TestConditionalLogLikelihoodValidatedScore:
    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return generate_normal_data_classification(DATA_SIZE // 5, seed=SEED)

    @pytest.fixture
    def base_model(self, df: pd.DataFrame) -> GaussianNaiveBayes:
        X = df.drop(columns=[TRUE_CLASS_LABEL])
        y = df[TRUE_CLASS_LABEL]
        model = GaussianNaiveBayes(seed=SEED)
        model.fit(X, y)
        return model

    @pytest.fixture
    def score(self, df: pd.DataFrame) -> ConditionalLogLikelihoodValidatedScore:
        return ConditionalLogLikelihoodValidatedScore(
            df=df,
            target=TRUE_CLASS_LABEL,
            model_class=GaussianNaiveBayes,
            test_ratio=0.2,
            k=2,
            seed=SEED,
        )

    def test_init_raises_if_target_missing(self, df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="missing_target"):
            ConditionalLogLikelihoodValidatedScore(
                df=df,
                target="missing_target",
                model_class=GaussianNaiveBayes,
                seed=SEED,
            )

    def test_init_raises_if_target_has_single_value(self, df: pd.DataFrame) -> None:
        single_class_df = df.copy()
        single_class_df[TRUE_CLASS_LABEL] = "class1"

        with pytest.raises(ValueError, match="requires at least two target values"):
            ConditionalLogLikelihoodValidatedScore(
                df=single_class_df,
                target=TRUE_CLASS_LABEL,
                model_class=GaussianNaiveBayes,
                seed=SEED,
            )

    def test_holdout_split_sizes_match_ratio(
        self,
        score: ConditionalLogLikelihoodValidatedScore,
        df: pd.DataFrame,
    ) -> None:
        total = len(df)
        train_len = len(score._training_data_holdout)
        test_len = len(score._test_data_holdout)

        assert train_len + test_len == total
        assert test_len == int(round(total * 0.2))

    def test_holdout_split_is_stratified(
        self,
        score: ConditionalLogLikelihoodValidatedScore,
        df: pd.DataFrame,
    ) -> None:
        full_classes = set(df[TRUE_CLASS_LABEL].unique())
        train_classes = set(score._training_data_holdout[TRUE_CLASS_LABEL].unique())
        test_classes = set(score._test_data_holdout[TRUE_CLASS_LABEL].unique())

        assert train_classes == full_classes
        assert test_classes == full_classes

    def test_stratified_splits_are_deterministic_for_same_seed(
        self,
        df: pd.DataFrame,
    ) -> None:
        score_a = ConditionalLogLikelihoodValidatedScore(
            df=df,
            target=TRUE_CLASS_LABEL,
            model_class=GaussianNaiveBayes,
            test_ratio=0.2,
            k=2,
            seed=SEED,
        )
        score_b = ConditionalLogLikelihoodValidatedScore(
            df=df,
            target=TRUE_CLASS_LABEL,
            model_class=GaussianNaiveBayes,
            test_ratio=0.2,
            k=2,
            seed=SEED,
        )

        assert score_a._training_data_holdout.equals(score_b._training_data_holdout)
        assert score_a._test_data_holdout.equals(score_b._test_data_holdout)

        cv_a = score_a._get_cv_splits()
        cv_b = score_b._get_cv_splits()
        assert len(cv_a) == len(cv_b) == 2

        for (a_train, a_test), (b_train, b_test) in zip(cv_a, cv_b):
            assert a_train.equals(b_train)
            assert a_test.equals(b_test)

    def test_cv_splits_are_stratified(
        self,
        score: ConditionalLogLikelihoodValidatedScore,
    ) -> None:
        full_classes = set(score._training_data_holdout[TRUE_CLASS_LABEL].unique())

        for train_df, test_df in score._get_cv_splits():
            assert set(train_df[TRUE_CLASS_LABEL].unique()) == full_classes
            assert set(test_df[TRUE_CLASS_LABEL].unique()) == full_classes

    def test_cv_test_fold_sizes_are_balanced(
        self,
        score: ConditionalLogLikelihoodValidatedScore,
    ) -> None:
        fold_test_sizes = [len(test_df) for _, test_df in score._get_cv_splits()]

        assert len(fold_test_sizes) > 0
        assert max(fold_test_sizes) - min(fold_test_sizes) <= 1

    def test_has_variables(self, score: ConditionalLogLikelihoodValidatedScore) -> None:
        assert score.has_variables([TRUE_CLASS_LABEL, "a"])
        assert score.has_variables("a")
        assert not score.has_variables(["a", "not_a_feature"])

    def test_compatible_bn(self, score: ConditionalLogLikelihoodValidatedScore) -> None:
        assert score.compatible_bn(
            pbn.GaussianNetwork([TRUE_CLASS_LABEL, "a", "b", "c"])
        )
        assert not score.compatible_bn(
            pbn.GaussianNetwork([TRUE_CLASS_LABEL, "a", "b", "x"])
        )

    def test_data_returns_original_dataframe(
        self, score: ConditionalLogLikelihoodValidatedScore, df: pd.DataFrame
    ) -> None:
        assert score.data() is df

    def test_model_with_variable_evidence_updates_parents(
        self,
        score: ConditionalLogLikelihoodValidatedScore,
        base_model: GaussianNaiveBayes,
    ) -> None:
        candidate = score._model_with_variable_evidence(
            base_model, variable="c", evidence=[TRUE_CLASS_LABEL, "a"]
        )

        assert set(candidate.parents("c")) == {TRUE_CLASS_LABEL, "a"}
        assert set(base_model.parents("c")) == {TRUE_CLASS_LABEL}

    def test_local_score_matches_manual_cv_accumulation(
        self,
        score: ConditionalLogLikelihoodValidatedScore,
        base_model: GaussianNaiveBayes,
    ) -> None:
        variable = "c"
        evidence = [TRUE_CLASS_LABEL, "a"]
        candidate = score._model_with_variable_evidence(base_model, variable, evidence)

        expected = 0.0
        for train_df, test_df in score._get_cv_splits():
            expected += score._conditional_log_likelihood(candidate, train_df, test_df)

        actual = score.local_score(base_model, variable, evidence)
        assert actual == pytest.approx(expected)

    def test_vlocal_score_matches_holdout_cll(
        self,
        score: ConditionalLogLikelihoodValidatedScore,
        base_model: GaussianNaiveBayes,
    ) -> None:
        variable = "c"
        evidence = [TRUE_CLASS_LABEL, "a"]
        candidate = score._model_with_variable_evidence(base_model, variable, evidence)

        expected = score._conditional_log_likelihood(
            candidate,
            score._training_data_holdout,
            score._test_data_holdout,
        )
        actual = score.vlocal_score(base_model, variable, evidence)
        assert actual == pytest.approx(expected)

    def test_local_score_node_type_matches_manual_cv_accumulation(
        self,
        score: ConditionalLogLikelihoodValidatedScore,
        base_model: GaussianNaiveBayes,
    ) -> None:
        variable = "c"
        evidence = [TRUE_CLASS_LABEL, "a"]
        variable_type = base_model.node_types()[variable]
        candidate = score._model_with_variable_evidence(base_model, variable, evidence)
        candidate.set_node_type(variable, variable_type)

        expected = 0.0
        for train_df, test_df in score._get_cv_splits():
            expected += score._conditional_log_likelihood(candidate, train_df, test_df)

        actual = score.local_score_node_type(
            base_model,
            variable_type,
            variable,
            evidence,
        )
        assert actual == pytest.approx(expected)

    def test_vlocal_score_node_type_matches_holdout_cll(
        self,
        score: ConditionalLogLikelihoodValidatedScore,
        base_model: GaussianNaiveBayes,
    ) -> None:
        variable = "c"
        evidence = [TRUE_CLASS_LABEL, "a"]
        variable_type = base_model.node_types()[variable]
        candidate = score._model_with_variable_evidence(base_model, variable, evidence)
        candidate.set_node_type(variable, variable_type)

        expected = score._conditional_log_likelihood(
            candidate,
            score._training_data_holdout,
            score._test_data_holdout,
        )
        actual = score.vlocal_score_node_type(
            base_model,
            variable_type,
            variable,
            evidence,
        )
        assert actual == pytest.approx(expected)

    def test_to_pandas_with_dataframe(self, df: pd.DataFrame) -> None:
        out = ConditionalLogLikelihoodValidatedScore._to_pandas(df)
        assert out is df

    def test_to_pandas_with_to_pandas_compatible_object(self, df: pd.DataFrame) -> None:
        class _DummyFrame:
            def __init__(self, frame: pd.DataFrame) -> None:
                self._frame = frame

            def to_pandas(self) -> pd.DataFrame:
                return self._frame

        wrapped = _DummyFrame(df)
        out = ConditionalLogLikelihoodValidatedScore._to_pandas(wrapped)
        assert out is df

    def test_to_pandas_raises_for_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            ConditionalLogLikelihoodValidatedScore._to_pandas(object())


class TestAccuracyScore:
    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return generate_normal_data_classification(DATA_SIZE // 5, seed=SEED)

    @pytest.fixture
    def base_model(self, df: pd.DataFrame) -> GaussianNaiveBayes:
        x = df.drop(columns=[TRUE_CLASS_LABEL])
        y = df[TRUE_CLASS_LABEL]
        model = GaussianNaiveBayes(seed=SEED)
        model.fit(x, y)
        return model

    @pytest.fixture
    def score(self, df: pd.DataFrame) -> AccuracyScore:
        return AccuracyScore(
            df=df,
            target=TRUE_CLASS_LABEL,
            model_class=GaussianNaiveBayes,
            test_ratio=0.2,
            seed=SEED,
        )

    def test_has_variables(self, score: AccuracyScore) -> None:
        assert score.has_variables([TRUE_CLASS_LABEL, "a"])
        assert score.has_variables("a")
        assert not score.has_variables(["a", "not_a_feature"])

    def test_compatible_bn(self, score: AccuracyScore) -> None:
        assert score.compatible_bn(
            pbn.GaussianNetwork([TRUE_CLASS_LABEL, "a", "b", "c"])
        )
        assert not score.compatible_bn(
            pbn.GaussianNetwork([TRUE_CLASS_LABEL, "a", "b", "x"])
        )

    def test_score_returns_accuracy_between_zero_and_one(
        self,
        score: AccuracyScore,
        base_model: GaussianNaiveBayes,
    ) -> None:
        value = score.score(base_model)
        assert 0.0 <= value <= 1.0

    def test_local_score_node_type_matches_manual_accuracy(
        self,
        score: AccuracyScore,
        base_model: GaussianNaiveBayes,
    ) -> None:
        variable = "c"
        evidence = [TRUE_CLASS_LABEL, "a"]
        variable_type = base_model.node_types()[variable]

        candidate = score._model_with_variable_evidence(base_model, variable, evidence)
        candidate.set_node_type(variable, variable_type)
        expected = score._accuracy(candidate)

        actual = score.local_score_node_type(
            base_model,
            variable_type,
            variable,
            evidence,
        )
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize("score_class", [F1Score, AUCScore])
class TestF1AndAUCScore:
    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return generate_normal_data_classification(DATA_SIZE // 5, seed=SEED)

    @pytest.fixture
    def base_model(self, df: pd.DataFrame) -> GaussianNaiveBayes:
        x = df.drop(columns=[TRUE_CLASS_LABEL])
        y = df[TRUE_CLASS_LABEL]
        model = GaussianNaiveBayes(seed=SEED)
        model.fit(x, y)
        return model

    @pytest.fixture
    def score(
        self, df: pd.DataFrame, score_class: type[AccuracyScore]
    ) -> AccuracyScore:
        return score_class(
            df=df,
            target=TRUE_CLASS_LABEL,
            model_class=GaussianNaiveBayes,
            test_ratio=0.2,
            seed=SEED,
        )

    def test_score_returns_value_between_zero_and_one(
        self,
        score: AccuracyScore,
        base_model: GaussianNaiveBayes,
    ) -> None:
        value = score.score(base_model)
        assert 0.0 <= value <= 1.0

    def test_local_score_node_type_matches_manual_metric(
        self,
        score: AccuracyScore,
        base_model: GaussianNaiveBayes,
    ) -> None:
        variable = "c"
        evidence = [TRUE_CLASS_LABEL, "a"]
        variable_type = base_model.node_types()[variable]

        candidate = score._model_with_variable_evidence(base_model, variable, evidence)
        candidate.set_node_type(variable, variable_type)
        expected = score._accuracy(candidate)

        actual = score.local_score_node_type(
            base_model,
            variable_type,
            variable,
            evidence,
        )
        assert actual == pytest.approx(expected)
