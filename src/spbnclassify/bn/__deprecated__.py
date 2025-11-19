# # TODO: Reimplement with the new GaussianBayesianNetwork
# # Gaussian Bayesian Network Multinet class for Anomaly Detection
# import pickle
# import re
# import warnings
# from copy import deepcopy

# import matplotlib.pyplot as plt
# import numpy as np
# from anytree import Node, RenderTree
# from sklearn.cluster import KMeans
# from sklearn.pipeline import Pipeline

# # To suppress warnings from np.exp
# warnings.filterwarnings("ignore")


# def extract_cluster_index(cluster_name):
#     """Returns the cluster index from the cluster name

#     Args:
#         cluster_name (str): string with the r_<k> format

#     Returns:
#         int: Returns the cluster index
#     """
#     p = re.compile(r"\d+")
#     m = p.search(cluster_name)
#     if m:
#         k = int(m.group())
#     else:
#         k = None
#     return k


# class GaussianBayesianNetworkMixture:
#     def __init__(
#         self,
#         n_components=2,
#         covariance_type="full",
#         tol=1e-3,
#         reg_covarfloat=1e-6,
#         max_iter=10,
#         n_init=1,
#         init_params="kmeans",
#         weights_init=None,
#         # means_init=None,
#         precisions_init=None,
#         seed=None,
#         warm_start=False,
#         verbose=0,
#         verbose_interval=10,
#         feature_names=[],
#     ):
#         self.n_components = n_components
#         self.weights_ = np.empty(n_components)
#         self.best_weights_ = None
#         # if weights_init is not None:
#         #     assert len(weights_init) == n_components
#         #     assert np.isclose(np.sum(weights_init), 1.0)
#         #     self.weights_ = weights_init
#         # else:
#         #     # if init_params == "kmeans":
#         #     self.weights_ = np.ones(n_components) / n_components

#         self.max_iter = max_iter
#         # self.means_ = None
#         # self.covariances_ = None
#         self.tol = tol
#         # self.precisions_ = None
#         # self.precisions_cholesky_ = None
#         self.converged_ = False
#         self.n_iter_ = 0
#         self.lower_bound_ = -np.inf

#         self.feature_names_in_ = feature_names
#         self.n_features_in_ = len(feature_names)
#         self.training_data = None
#         self.bn_type = GaussianBayesianNetwork

#         self.root = Node("Z")
#         self.bn_leaves = {}
#         self.best_bn_leaves = None

#         for k in range(self.n_components):
#             self.bn_leaves[k] = Node(k, parent=self.root, bn=None)

#         self.model_info = {
#             "structure_algorithm": "HC",
#         }
#         self.train_info = {
#             "trained": False,
#         }

#     def __str__(self):
#         return f"Mixture of Gaussian Bayesian Networks with {self.n_components} components and {self.n_features_in_} features"

#     def get_info(self):
#         """Returns the model information

#         Returns:
#             dict: dictionary with the model information
#         """
#         return self.model_info

#     # RFE: this right now uses GBN in each fit to learn the parameters of the model... Ideally, it should be updated with a quicker Bayesian method. Probably updating structure and parameters step by step
#     def fit(
#         self,
#         data,
#         score="bic",
#         arc_blacklist=[],
#         num_folds=5,
#         max_iters=2147483647,
#         seed=None,
#     ):
#         # 1. Initialize the μk’s, σk’s and πk’s and evaluate the log-likelihood with these parameters.
#         # 2. E-step: Evaluate the posterior probabilities γZi(k) using the current values of the μk’s and σk’s with equation (2)
#         # 3. M-step: Estimate new parameters μk^, σ2k^ and πk^ with the current values of γZi(k) using equations (3), (4) and (5).
#         # 4. Evaluate the log-likelihood with the new parameter estimates. If the log-likelihood has changed by less than some small ϵ, stop. Otherwise, go back to step 2.
#         self.training_data = data
#         # Initial light clustering
#         clusterer = Pipeline(
#             steps=[
#                 # ("preprocessor", preprocessor),
#                 (
#                     "classifier",
#                     KMeans(
#                         n_clusters=self.n_components,
#                         init="k-means++",
#                         n_init=10,
#                         max_iter=300,
#                         tol=self.tol,
#                         verbose=0,
#                         random_state=seed,
#                         copy_x=True,
#                         algorithm="lloyd",
#                     ),
#                 ),
#             ]
#         )
#         self.training_data["cluster_label"] = clusterer.fit_predict(
#             self.training_data[self.feature_names_in_]
#         )

#         self.weights_ = self.training_data["cluster_label"].value_counts(normalize=True)
#         print(
#             "Number of clusters:\t"
#             + str(np.unique(self.training_data["cluster_label"], return_counts=True))
#         )
#         self.bn_leaves = {}
#         for k in range(self.n_components):
#             bn = self.bn_type(nodes=self.feature_names_in_)
#             bn.fit(
#                 self.training_data[self.training_data["cluster_label"] == k],
#                 score=score,
#                 arc_blacklist=arc_blacklist,
#                 num_folds=num_folds,
#                 max_iters=max_iters,
#                 seed=seed,
#             )
#             self.bn_leaves[k] = Node(k, parent=self.root, bn=bn)

#         self.lower_bound_ = self.slogl(self.training_data)
#         print("\nINITIALIZATION:")
#         print("Cluster weights:\t" + str(self.weights_))
#         print("Initial log-likelihood:", self.lower_bound_)
#         # NOTE: No guaranteed convergence
#         for self.n_iter_ in range(self.max_iter):
#             print(f"\nStarting iteration {self.n_iter_}")
#             self.training_data = self.Estep(self.training_data)
#             print("Estep done")
#             self.Mstep(
#                 self.training_data,
#                 score=score,
#                 arc_blacklist=arc_blacklist,
#                 num_folds=num_folds,
#                 max_iters=max_iters,
#                 seed=seed,
#             )
#             print("Mstep done")
#             print("Cluster weights:\t" + str(self.weights_))

#             # Current log-likelihood
#             lb = self.slogl(self.training_data)
#             lb_diff = lb - self.lower_bound_
#             print("Log-likelihood:", lb)
#             if lb_diff < 0:
#                 print(f"ERROR: Log-likelihood decreased {lb_diff} < 0")
#             else:
#                 # NOTE: We save the best model
#                 self.lower_bound_ = lb
#                 self.best_weights_ = self.weights_.copy()
#                 self.best_bn_leaves = deepcopy(self.bn_leaves)

#                 print("Log-likelihood INCREASED:", f"{lb_diff} >= 0")
#                 if lb_diff <= self.tol:
#                     self.converged_ = True
#                     print("Converged")
#                     break
#                 else:  # lb_diff > self.tol
#                     print("Not converged")

#         # Finish with best model
#         self.bn_leaves = self.best_bn_leaves
#         self.weights_ = self.best_weights_
#         self.model_info.update({"anomaly_score": score})
#         if self.bn_leaves is not None:
#             self.train_info.update(
#                 {
#                     "trained": all(
#                         [
#                             self.bn_leaves[k].bn.fitted()
#                             for k in range(self.n_components)
#                         ]
#                     ),
#                     "data_columns": self.feature_names_in_,
#                 }
#             )

#     def calculate_cluster_responsibility(self, data):
#         # Calculates the responsibility of each component r_k = p(z_i = k|x)
#         if self.bn_leaves is not None and self.weights_ is not None:
#             for k in range(self.n_components):
#                 # data[f"r_{k}"] = self.weights_[k] * np.exp(self.bn_leaves[k].bn.logl(data))
#                 # We calculate the logl of each component

#                 data[f"logl_{k}"] = self.bn_leaves[k].bn.logl(data)
#             for k in range(self.n_components):
#                 # We calculate the responsibility of each component
#                 data[f"r_{k}"] = 0

#                 for j in range(self.n_components):
#                     # This version doesn't work because of numerical problems
#                     # data[f"r_{k}"] += np.exp(
#                     #     np.log(self.weights_[j])
#                     #     - np.log(self.weights_[k])
#                     #     + data[f"logl_{j}"]
#                     #     - data[f"logl_{k}"]
#                     # )
#                     data[f"r_{k}"] += self.weights_[j] * np.exp(
#                         data[f"logl_{j}"] - data[f"logl_{k}"]
#                     )
#                 data[f"r_{k}"] = self.weights_[k] / data[f"r_{k}"]

#         # data["sum_r_k"] = data[[f"r_{k}" for k in range(self.n_components)]].sum(axis=1)
#         # for k in range(self.n_components):
#         #     data[f"r_{k}"] = data[f"r_{k}"] / data["sum_r_k"]

#         # Or rethink GBN's standardized logl?

#         data.drop([f"logl_{k}" for k in range(self.n_components)], axis=1, inplace=True)
#         return data

#     def assign_cluster(self, data):
#         # Reassigns cluster label to the cluster "r_{k}" with more probability
#         data["cluster_label"] = data[
#             [f"r_{k}" for k in range(self.n_components)]
#         ].idxmax(axis=1)

#         # Extracts the index from the cluster name
#         data["cluster_label"] = data["cluster_label"].apply(extract_cluster_index)
#         return data

#     def Estep(self, data):
#         "Perform an Expectation step: Calculates the current responsibility of each component r_k = p(z_i = k|x)"
#         data = self.calculate_cluster_responsibility(data)
#         data = self.assign_cluster(data)
#         return data

#     def Mstep(
#         self,
#         data,
#         score="bic",
#         arc_blacklist=[],
#         num_folds=5,
#         max_iters=2147483647,
#         seed=None,
#     ):
#         "Perform an Maximization step: Updates the weights and the model parameters and structure"
#         # RFE: parallelize
#         if self.bn_leaves is not None and self.weights_ is not None:
#             for k in range(self.n_components):
#                 cluster_mask = data["cluster_label"] == k
#                 self.weights_[k] = data[f"r_{k}"].mean()
#                 # NOTE: We always reset the structure
#                 bn = self.bn_type(nodes=self.feature_names_in_)
#                 bn.fit(
#                     data[cluster_mask],
#                     score=score,
#                     arc_blacklist=arc_blacklist,
#                     num_folds=num_folds,
#                     max_iters=max_iters,
#                     seed=seed,
#                 )
#                 self.bn_leaves[k].bn = bn

#     def logl(self, data):  # IDEA: test with standardized log-likelihood
#         # Calculates the log-likelihood of the data
#         data = self.Estep(data)
#         if self.bn_leaves is not None and self.weights_ is not None:
#             for k in range(self.n_components):
#                 cluster_mask = data["cluster_label"] == k
#                 if not data[cluster_mask].empty:
#                     data.loc[cluster_mask, "log_likelihood"] = self.bn_leaves[
#                         k
#                     ].bn.logl(data[cluster_mask]) + np.log(self.weights_[k])
#             return data["log_likelihood"]
#         else:
#             return np.nan

#     def slogl(self, data):
#         return self.logl(data).sum()

#     def _get_joint_gaussian(self):
#         """Returns the parameters of the equivalent Gaussian joint distribution

#         Returns:
#             tuple: Parameters of the model
#         """
#         print("self.n_components_", self.n_components)
#         print("self.weights_: ", self.weights_)
#         if self.bn_leaves is not None:
#             for k in range(self.n_components):
#                 gaussian_mean, gaussian_cov = self.bn_leaves[k].bn._get_joint_gaussian()
#                 print("\nComponent: ", k)
#                 print("gaussian_mean:\n", gaussian_mean)
#                 print("gaussian_cov:\n", gaussian_cov)

#     def explain(self, data):
#         if self.bn_leaves is not None and self.weights_ is not None:
#             for k in range(self.n_components):
#                 cluster_mask = data["cluster_label"] == k
#                 if not data[cluster_mask].empty:
#                     data[cluster_mask] = self.bn_leaves[k].bn.explain(
#                         data[cluster_mask]
#                     )
#                     data.loc[cluster_mask, "log_likelihood"] += np.log(self.weights_[k])
#         return data

#     def show(self):
#         for pre, _, node in RenderTree(self.root):
#             print("%s%s" % (pre, node.name))
#         print("")
#         if self.bn_leaves is not None and self.weights_ is not None:
#             for node in self.bn_leaves.values():
#                 _, ax = plt.subplots()
#                 ax.set_title(node.name)
#                 print(f"{node.name} NODE")
#                 print(f"Weight: {self.weights_[node.name]}")
#                 node.bn.show(ax=ax, file_name=None)

#     def save(self, filename):
#         with open(filename, "wb") as f:
#             pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

#     @classmethod
#     def load(cls, filename):
#         with open(filename, "rb") as f:
#             return pickle.load(f)

# RFE: Añadir CPT calculation for DiscreteFactor
# from rutile_ai.workflow.data_preparation.variables.nfstream_cols import (
#     categorical_variables,
# )
# def bn_cpt(cpd):
#     """Calculates the conditional probability table of a discrete node given its parents.

#     Args:
#         cpd (pbn.Factor): Conditional Probability Distribution

#     Returns:
#         pd.DataFrame: Dataframe with the conditional probability table
#     """
#     cpt = None
#     if cpd.type().__str__() == "DiscreteFactor":
#         node = cpd.variable()
#         parents = cpd.evidence()
#         variables = [node] + parents
#         categories = [categorical_variables[v] for v in variables]

#         # Cartesian product of all possible categories
#         categories_cp = list(itertools.product(*categories))
#         # Dataframe with all possible category combinations
#         df = pd.DataFrame(categories_cp, columns=variables, dtype="category")

#         # node variable is a "title" column of the dataframe
#         # its subcolumns are the possible categories
#         # columns = pd.MultiIndex.from_product(
#         #     [[node], categorical_variables[node]])
#         columns = pd.Index(categorical_variables[node], name=node)

#         if not parents:
#             index = None
#             cpt = [np.exp(cpd.logl(df))]  # We treat cpt as a row array
#         else:
#             # Cartesian product of all categories
#             index = pd.MultiIndex.from_product(
#                 [categorical_variables[p] for p in parents], names=parents
#             )
#             cpt = np.exp(cpd.logl(df)).reshape(
#                 index.shape[0], columns.shape[0], order="f"
#             )  # CPT is reshaped as table

#         cpt = pd.DataFrame(cpt, index=index, columns=columns)
#     return cpt

# TODO: Review usability for HybridGaussianBayesianNetwork
# def bn_smooth(self, cpt, data):
#     """Smooths the CPDs of a Bayesian Network.
#     For discrete variables, the CPTs with learnt zero probabilities we use Laplace Estimation.

#     Args:
#         bn (pbn.BayesianNetworkBase): Bayesian Network
#         cpt (pd.DataFrame): Conditional Probability Table
#         data (pd.DataFrame): Training data for the Bayesian Network

#     Returns:
#         tuple: (bn, cpt, data)
#     """
#     MIN_VARIANCE = 1e-4
#     # IDEA: Might rethink variance assignment -> Try smallest when normalized log prob
#     # MIN_VARIANCE = np.nextafter(0, 1) # This makes the log-likelihood explode
#     modified_nodes = []
#     for node in self.nodes():
#         node_type = self.node_type(node).__str__()
#         cpd = self.cpd(node)
#         parents = cpd.evidence()

#         # if node_type == "DiscreteFactor":
#         #     # Mask for rows with any zero probability
#         #     incomplete_cpt_mask = cpt[node].apply(np.all, axis=1) is False

#         #     index = cpt[node][incomplete_cpt_mask].index
#         #     columns = cpt[node][incomplete_cpt_mask].columns

#         #     # Depending on if the discrete node has parents
#         #     if not parents:
#         #         variables = [columns.name]
#         #         categories_cp = columns.values
#         #     else:
#         #         variables = [columns.name] + index.names
#         #         categories = [columns.values, index]
#         #         # Cartesian product of all possible categories
#         #         categories_cp = list(itertools.product(*categories))
#         #         categories_cp = list(map(lambda x: (x[0],) + x[1], categories_cp))

#         #     # Dataframe with all possible category combinations for the missing observations
#         #     df = pd.DataFrame(categories_cp, columns=variables)
#         #     # Laplace estimation
#         #     df = pd.concat([data[variables], df], ignore_index=True).astype("category")
#         #     cpd.fit(df)

#         #     # ! This fails when training data has less categories than expected
#         #     cpt[node] = bn_cpt(bn.cpd(node))  # Update CPT
#         #     print(f"Discrete node modified for node: {node}")

#         if (
#             node_type == "LinearGaussianFactor"
#         ):  # BUG: doesn't work when node_type == CLinearGaussianFactor
#             # if len(parents) == 0:  # NOTE: No parents, as with discrete parents it fails
#             if cpd.variance == 0:  # Case of Gaussian with only one unique value
#                 cpd.variance = MIN_VARIANCE
#                 modified_nodes.append(node)
#                 # Remove variable
#                 # bn.remove_node(node)
#                 # print(f"0 variance node removed: {node}")

# NOTE: Code for HybridGaussianBayesianNetwork
#         # elif node_type == 'CKDEFactor':
#         #     pass # RFE: check in case CKDE not fitted all parameters and how to calculate f_k
#     if modified_nodes:
#         print(f"Variance modified for nodes: {modified_nodes}")
#     return self, cpt

# RFE: Añadir calculation for DiscreteFactor and (HCLG/HCKDE)
# if node_type == "DiscreteFactor":
#     categories = categorical_variables[node]
#     df = pd.DataFrame({node: pd.Categorical(categories, categories=categories)})

#     for p in parents:  # In case of having discrete parents
#         p_type = bn.node_type(p).__str__()
#         if p_type == "DiscreteFactor":
#             p_categories = categorical_variables[p]
#             df = df.merge(
#                 pd.DataFrame(
#                     {p: pd.Categorical(p_categories, categories=p_categories)}
#                 ),
#                 how="cross",
#             )  # Cartesian product

#     df["log_likelihood"] = cpd.logl(df)
#     ell = df["log_likelihood"].max()

# else:  # Continuous CPD
# discrete_parents = [
#     p for p in parents if self.node_type(p).__str__() == "DiscreteFactor"
# ]
# if not discrete_parents:


# NOTE: Optimization examples
# BONUS: OPTIMIZATION EXAMPLES
# LOCAL OPTIMIZATION
# Initial guess
# x0 = np.array([(data[node].min() + data[node].max())/2, (data[parents[0]].min() + data[parents[0]].max())])
# print(- _logl_objective_function(x0, bn, node))
# res = minimize(lambda x: - _logl_objective_function(x, bn, node), x0, bounds=bounds)# 0.7s (local maxima), it depends on x0

# GLOBAL OPTIMIZATION
# res = differential_evolution(lambda x: - _logl_objective_function(x, bn, node), bounds=bounds) # 0.5s stochastic <- Best trade-off between best result and least time
# res = dual_annealing(lambda x: - _logl_objective_function(x, bn, node), bounds=bounds) # 6.5s
# res = shgo(lambda x: - _logl_objective_function(x, bn, node), bounds=bounds) # 0.1s but not the best solution
# res = direct(lambda x: - _logl_objective_function(x, bn, node), bounds=bounds) # 2.5s
