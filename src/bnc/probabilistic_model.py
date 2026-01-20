import itertools

import numpy as np
import pyarrow as pa
import pybnesian as pbn
import scipy.special

from ..utils.constants import TRUE_CLASS_LABEL

# RFE: from helpers.data import TRUE_CLASS_LABEL
ISS_PRIOR_DISCRETE = 3  # Dirichlet prior for discrete CPDs
ISS_PRIOR_NMM = 3  # Dirichlet prior for Normal Mixture CPDs

NUM_CLASSES = 3
DISCRETE_CATEGORIES_DICT = {
    TRUE_CLASS_LABEL: ["class" + str(i) for i in range(NUM_CLASSES)],
}


# region FixedDiscreteFactor


class FixedDiscreteFactor(pbn.Factor):
    """
    A class representing a fixed discrete factor in a probabilistic Bayesian network.
    Attributes:
        variable (str): The variable for which the factor is defined.
        evidence (list[str]): A list of evidence variables that the factor depends on.
        variable_values (list[str]): A list of possible values for the variable.
        evidence_values (list[list[str]]): A list of lists, where each sublist contains possible values for each evidence variable.
        discrete_configs (list[tuple[str, ...]]): A list of tuples representing all possible configurations of evidence values.
        logprob (np.ndarray): A 2D array of log probabilities for each configuration of evidence values and variable values.
    Methods:
        new_random_cpd(cls, variable, evidence):
            Generate a new random conditional probability distribution (CPD) for a given variable and its evidence variables.
        data_type():
            Return the data type of the factor as a dictionary with int8 keys and utf8 values.
        fitted():
            Return True indicating that the factor is fitted.
        logl(df):
            Compute the log-likelihood of the given DataFrame based on the factor.
        slogl(df):
            Compute the sum of log-likelihoods of the given DataFrame based on the factor.
        sample(n, evidence, seed):
            Generate samples from the factor given the number of samples, evidence, and a random seed.
        type():
            Return the type of the factor as FixedDiscreteFactorType.
        __getstate_extra__():
            Return extra state information for serialization.
        __setstate_extra__(extra):
            set extra state information during deserialization.
    """

    def __init__(
        self,
        variable: str,
        evidence: list[str],
        variable_values: list[str],
        evidence_values: list[list[str]],
        discrete_configs: list[tuple[str, ...]],
        logprob: np.ndarray,
    ):
        pbn.Factor.__init__(self, variable, evidence)
        self.variable_values = variable_values
        self.evidence_values = evidence_values
        self.discrete_configs = discrete_configs
        self.logprob = logprob

    def __str__(self) -> str:
        return f"FixedDiscreteFactor(variable={self.variable()}, evidence={self.evidence()})"

    @classmethod
    def new_random_cpd(cls, variable, evidence):
        """
        Generate a new random conditional probability distribution (CPD) for a given variable
        and its evidence variables.
        Args:
            variable (str): The variable for which the CPD is being generated.
            evidence (list of str): A list of evidence variables that the CPD depends on.
        Returns:
            FixedDiscreteFactor: An object representing the CPD with the specified variable,
                                evidence, categories, configurations, and log probabilities.
        """

        cats = [DISCRETE_CATEGORIES_DICT[n] for n in evidence]
        discrete_configs = list(itertools.product(*cats))

        num_var_cats = len(DISCRETE_CATEGORIES_DICT[variable])
        num_configs = len(discrete_configs)

        logprob = np.empty((num_configs, num_var_cats), dtype=float)

        for i in range(num_configs):
            logprob[i, :] = np.log(
                np.random.dirichlet([ISS_PRIOR_DISCRETE] * num_var_cats)
            )

        return FixedDiscreteFactor(
            variable,
            evidence,
            DISCRETE_CATEGORIES_DICT[variable],
            cats,
            discrete_configs,
            logprob,
        )

    def data_type(self) -> dict:
        """
        Return the data type of the factor as a dictionary with int8 keys and utf8 values.
        Returns:
            dict: A dictionary with int8 keys and utf8 values.
        """
        return pa.dictionary(pa.int8(), pa.utf8())

    def fitted(self):
        return True

    def logl(self, df):
        df_pandas = df.to_pandas()
        ll = np.empty((df_pandas.shape[0],), dtype=float)

        for i, config in enumerate(self.discrete_configs):
            df_config = df_pandas

            # Filter by discrete configuration
            for dv, dc in zip(self.evidence(), config):
                df_config = df_config.where(df_config[dv] == dc)

            for j, var_value in enumerate(self.variable_values):
                df_value = df_config.where(df_config[self.variable()] == var_value)

                # Filtered rows are NaN
                not_null_indices = df_value.notnull().all(axis=1)

                ll[not_null_indices] = self.logprob[i, j]

        return ll

    def slogl(self, df):
        return self.logl(df).sum()

    def sample(self, n, evidence, seed):
        np.random.seed(seed)
        s = np.full((n,), "", dtype="object")

        if evidence is not None:
            evidence_pandas = evidence.to_pandas()

            for i, config in enumerate(self.discrete_configs):
                ev_config = evidence_pandas
                # Filter by discrete configuration
                for dv, dc in zip(self.evidence(), config):
                    ev_config = ev_config.where(ev_config[dv] == dc)

                # Filtered rows are NaN
                not_null_indices = ev_config.notnull().all(axis=1)
                # Filter df
                ev_config = ev_config[not_null_indices]

                arr_variable_values = np.array(self.variable_values)

                s[not_null_indices] = arr_variable_values[
                    np.random.choice(
                        len(self.variable_values),
                        p=np.exp(self.logprob[i, :]),
                        size=ev_config.shape[0],
                    )
                ]
        else:
            arr_variable_values = np.array(self.variable_values)

            s = arr_variable_values[
                np.random.choice(
                    len(self.variable_values), p=np.exp(self.logprob[0, :]), size=n
                )
            ]

        return pa.array(s)

    def type(self):
        return pbn.DiscreteFactorType()

    def __getstate_extra__(self):
        return (
            self.variable_values,
            self.evidence_values,
            self.discrete_configs,
            self.logprob,
        )

    def __setstate_extra__(self, extra):
        self.variable_values = extra[0]
        self.evidence_values = extra[1]
        self.discrete_configs = extra[2]
        self.logprob = extra[3]


# endregion FixedDiscreteFactor


# region FixedCLG


class FixedCLG(pbn.Factor):
    """
    FixedCLG is a specialized factor representing a Conditional Linear Gaussian (CLG) distribution
    for use in Bayesian networks with both discrete and continuous evidence variables.
    This class models the conditional probability distribution (CPD) of a variable given a set of
    discrete and continuous parent variables. For each configuration of the discrete evidence,
    a separate Linear Gaussian CPD is maintained. The class provides methods for generating random
    CPDs, computing log-likelihoods, sampling, and serialization.
    Attributes:
        discrete_evidence (list): list of discrete parent variable names.
        continuous_evidence (list): list of continuous parent variable names.
        discrete_configs (list): list of all possible configurations (tuples) of discrete evidence.
        _lgs (dict): Mapping from discrete evidence configurations to LinearGaussianCPD instances.
    Methods:
        __init__(...): Initialize the FixedCLG with variable, evidence, discrete/continuous parents, configs, and CPDs.
        __str__(): Return a string representation of the FixedCLG.
        new_random_cpd(...): Class method to generate a random FixedCLG for a given variable and evidence.
        data_type(): Return the data type of the modeled variable (float64).
        fitted(): Return True, indicating the model is always considered fitted.
        logl(df): Compute the log-likelihood for each row in the given DataFrame.
        slogl(df): Compute the sum of log-likelihoods over the DataFrame.
        sample(n, evidence, seed): Generate samples from the distribution given evidence.
        type(): Return the type of the CPD (LinearGaussianCPDType).
        __getstate_extra__(): Return extra state for serialization.
        __setstate_extra__(extra): Restore extra state from serialization.
    """

    def __init__(
        self,
        variable,
        evidence,
        discrete_evidence,
        continuous_evidence,
        discrete_configs,
        lgs,
    ):
        pbn.Factor.__init__(self, variable, evidence)
        self.discrete_evidence = discrete_evidence
        self.continuous_evidence = continuous_evidence
        self.discrete_configs = discrete_configs
        self._lgs = lgs

    def __str__(self) -> str:
        return f"FixedCLG(variable={self.variable()}, evidence={self.evidence()})"

    @classmethod
    def new_random_cpd(cls, variable, discrete_evidence, continuous_evidence):
        """
        Create a new random Conditional Probability Distribution (CPD) for a given variable.
        This method generates a new random CPD for a specified variable, considering both discrete and continuous evidence.
        The CPD is modeled using a Linear Gaussian CPD for each configuration of discrete evidence.
        Args:
            variable (str): The variable for which the CPD is being created.
            discrete_evidence (list): A list of discrete evidence variables.
            continuous_evidence (list): A list of continuous evidence variables.
        Returns:
            FixedCLG: An instance of FixedCLG representing the CPD for the given variable.
        """

        evidence = discrete_evidence + continuous_evidence

        cats = [DISCRETE_CATEGORIES_DICT[n] for n in discrete_evidence]
        discrete_configs = list(itertools.product(*cats))

        lgs = {}
        for conf in discrete_configs:
            betas = np.empty((len(continuous_evidence) + 1,))
            betas[0] = np.random.normal(0, 2)
            betas[1:] = np.random.choice(
                [1, -1], size=len(continuous_evidence), p=[0.5, 0.5]
            ) * np.random.uniform(1, 5, size=len(continuous_evidence))

            var = 0.2 + np.random.chisquare(1)
            lgs[conf] = pbn.LinearGaussianCPD(variable, continuous_evidence, betas, var)

        return FixedCLG(
            variable,
            evidence,
            discrete_evidence,
            continuous_evidence,
            discrete_configs,
            lgs,
        )

    def data_type(self):
        return pa.float64()

    def fitted(self):
        return True

    def logl(self, df):
        df_pandas = df.to_pandas()
        ll = np.empty((df_pandas.shape[0],), dtype=float)

        for config in self.discrete_configs:
            df_config = df_pandas
            # Filter by discrete configuration
            for dv, dc in zip(self.discrete_evidence, config):
                df_config = df_config.where(df_config[dv] == dc)

            # Filtered rows are NaN
            not_null_indices = df_config.notnull().all(axis=1)
            # Filter df
            df_config = df_config[not_null_indices]

            ll[not_null_indices] = self._lgs[config].logl(df_config)

        return ll

    def slogl(self, df):
        return self.logl(df).sum()

    def sample(self, n, evidence, seed):
        np.random.seed(seed)
        s = np.empty((n,), dtype=float)

        if evidence is not None:
            evidence_pandas = evidence.to_pandas()

            for config in self.discrete_configs:
                ev_config = evidence_pandas
                # Filter by discrete configuration
                for dv, dc in zip(self.discrete_evidence, config):
                    ev_config = ev_config.where(ev_config[dv] == dc)

                # Filtered rows are NaN
                not_null_indices = ev_config.notnull().all(axis=1)
                # Filter df
                ev_config = ev_config[not_null_indices]

                s[not_null_indices] = self._lgs[config].sample(
                    ev_config.shape[0], ev_config, seed=seed + 328
                )
        else:
            s = self._lgs[()].sample(n, None, seed=seed + 389).to_numpy()

        return pa.array(s)

    def type(self):
        return pbn.LinearGaussianCPDType()

    def __getstate_extra__(self):
        return (
            self.discrete_evidence,
            self.continuous_evidence,
            self.discrete_configs,
            self._lgs,
        )

    def __setstate_extra__(self, extra):
        self.discrete_evidence = extra[0]
        self.continuous_evidence = extra[1]
        self.discrete_configs = extra[2]
        self._lgs = extra[3]


# endregion FixedCLG


# region NormalMixtureCPD


class NormalMixtureCPD(pbn.Factor):
    """
    A Conditional Probability Distribution (CPD) representing a mixture of linear Gaussian models
    conditioned on both discrete and continuous evidence variables. Each configuration of discrete
    evidence variables is associated with a mixture of linear Gaussian CPDs, with mixture weights
    (priors) and component parameters sampled randomly or provided.
    Parameters
    ----------
    variable : str
        The name of the dependent variable (the variable whose distribution is modeled).
    evidence : list of str
        list of all evidence variable names (both discrete and continuous).
    discrete_evidence : list of str
        Names of discrete evidence variables.
    continuous_evidence : list of str
        Names of continuous evidence variables.
    discrete_configs : list of tuple
        All possible configurations (Cartesian product) of discrete evidence variable values.
    priors : dict
        Dictionary mapping each discrete configuration to a vector of mixture weights (priors).
    lgs : dict
        Dictionary mapping each discrete configuration to a list of LinearGaussianCPD objects,
        representing the mixture components.
    Methods
    -------
    new_random_cpd(variable, discrete_evidence, continuous_evidence)
        Class method to generate a random NormalMixtureCPD for the given variable and evidence.
    data_type()
        Returns the data type of the modeled variable (float64).
    fitted()
        Returns True, indicating the CPD is always considered fitted.
    logl(df)
        Computes the log-likelihood for each row in the given DataFrame.
    slogl(df)
        Computes the sum of log-likelihoods over all rows in the DataFrame.
    sample(n, evidence, seed)
        Generates n samples from the CPD, optionally conditioned on provided evidence.
    type()
        Returns the type of the CPD (CKDEType).
    __getstate_extra__()
        Returns extra state for serialization.
    __setstate_extra__(extra)
        Restores extra state from serialization.
    Notes
    -----
    - This CPD supports both discrete and continuous evidence variables.
    - For each discrete configuration, a separate mixture of linear Gaussian models is maintained.
    - Used in probabilistic graphical models for hybrid Bayesian networks.
    """

    def __init__(
        self,
        variable,
        evidence,
        discrete_evidence,
        continuous_evidence,
        discrete_configs,
        priors,
        lgs,
    ):
        pbn.Factor.__init__(self, variable, evidence)
        self.discrete_evidence = discrete_evidence
        self.continuous_evidence = continuous_evidence
        self.discrete_configs = discrete_configs
        self._priors = priors
        self._lgs = lgs

    def __str__(self) -> str:
        return (
            f"NormalMixtureCPD(variable={self.variable()}, evidence={self.evidence()})"
        )

    @classmethod
    def new_random_cpd(cls, variable, discrete_evidence, continuous_evidence):
        evidence = discrete_evidence + continuous_evidence

        cats = [DISCRETE_CATEGORIES_DICT[n] for n in discrete_evidence]
        discrete_configs = list(itertools.product(*cats))

        priors = {}
        lgs = {}
        for conf in discrete_configs:
            num_components = np.random.choice([2, 3, 4], size=1, p=[0.4, 0.3, 0.3])[0]
            priors[conf] = np.random.dirichlet([ISS_PRIOR_NMM] * num_components)

            cpds = []
            for _ in range(num_components):
                betas = np.empty((len(continuous_evidence) + 1,))
                betas[0] = np.random.normal(0, 2)
                betas[1:] = np.random.choice(
                    [1, -1], size=len(continuous_evidence), p=[0.5, 0.5]
                ) * np.random.uniform(1, 5, size=len(continuous_evidence))

                var = 0.2 + np.random.chisquare(1)
                cpds.append(
                    pbn.LinearGaussianCPD(variable, continuous_evidence, betas, var)
                )

            lgs[conf] = cpds

        return NormalMixtureCPD(
            variable,
            evidence,
            discrete_evidence,
            continuous_evidence,
            discrete_configs,
            priors,
            lgs,
        )

    def data_type(self):
        return pa.float64()

    def fitted(self):
        return True

    def logl(self, df):
        df_pandas = df.to_pandas()
        ll = np.empty((df_pandas.shape[0],), dtype=float)

        for config in self.discrete_configs:
            df_config = df_pandas
            # Filter by discrete configuration
            for dv, dc in zip(self.discrete_evidence, config):
                df_config = df_config.where(df_config[dv] == dc)

            # Filtered rows are NaN
            not_null_indices = df_config.notnull().all(axis=1)
            # Filter df
            df_config = df_config[not_null_indices]

            # num_components = self._priors[config].shape[0]

            logpriors = np.log(self._priors[config])
            ll_matrix = np.tile(logpriors, (df_config.shape[0], 1)).T

            for i, lg in enumerate(self._lgs[config]):
                ll_matrix[i, :] += lg.logl(df_config)

            ll[not_null_indices] = scipy.special.logsumexp(ll_matrix, axis=0)

        return ll

    def slogl(self, df):
        return self.logl(df).sum()

    def sample(self, n, evidence, seed):
        np.random.seed(seed)
        s = np.empty((n,), dtype=float)

        if evidence is not None:
            evidence_pandas = evidence.to_pandas()

            for config in self.discrete_configs:
                ev_config = evidence_pandas
                # Filter by discrete configuration
                for dv, dc in zip(self.discrete_evidence, config):
                    ev_config = ev_config.where(ev_config[dv] == dc)

                # Filtered rows are NaN
                not_null_indices = ev_config.notnull().all(axis=1)
                # Filter df
                ev_config = ev_config[not_null_indices]

                priors = self._priors[config]
                num_components = priors.shape[0]
                component = np.random.choice(
                    num_components, p=priors, size=ev_config.shape[0]
                )

                s_config = np.empty((ev_config.shape[0],), dtype=float)

                ev_config.reset_index(inplace=True)

                for i, cpd in enumerate(self._lgs[config]):
                    component_size = np.sum(component == i)
                    evidence_df = ev_config.iloc[component == i, :]
                    s_config[component == i] = cpd.sample(
                        component_size, evidence_df, seed=seed + 357
                    ).to_numpy()

                s[not_null_indices] = s_config
        else:
            priors = self._priors[()]
            num_components = priors.shape[0]
            component = np.random.choice(num_components, p=priors, size=n)

            for i, cpd in enumerate(self._lgs[()]):
                component_size = np.sum(component == i)
                evidence_df = evidence.to_pandas().iloc[component == i, :]
                s[component == i] = cpd.sample(
                    component_size, evidence_df, seed=seed + 337
                ).to_numpy()

        return pa.array(s)

    def type(self):
        return pbn.CKDEType()

    def __getstate_extra__(self):
        return (
            self.discrete_evidence,
            self.continuous_evidence,
            self.discrete_configs,
            self._priors,
            self._lgs,
        )

    def __setstate_extra__(self, extra):
        self.discrete_evidence = extra[0]
        self.continuous_evidence = extra[1]
        self.discrete_configs = extra[2]
        self._priors = extra[3]
        self._lgs = extra[4]


# endregion NormalMixtureCPD
