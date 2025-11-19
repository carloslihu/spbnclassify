import re

import numpy as np
import pyagrum as gum
import pybnesian as pbn
import scipy

from . import BayesianNetwork


def bn_to_acronym(name: str) -> str:
    """
    Converts a Bayesian network name to its acronym.
    This function capitalizes words or uppercase abbreviations to form an acronym.
    For each word, if it is already uppercase, it is included as is; otherwise, only
    the first letter (capitalized) is used.
    Args:
        name (str): The name of the Bayesian network.
    Returns:
        str: The acronym generated from the input name.
    Example:
        >>> bn_to_acronym("SemiParametricBayesianNetwork")
        'SPBN'
    """

    words = re.findall(r"[A-Z][a-z]*|[A-Z]+(?![a-z])", name)
    acronym = "".join([w if w and w.isupper() else w[0].upper() for w in words if w])
    # Insert hyphen after G, SP, or KDE if acronym starts with one of these
    for prefix in ["G", "SP", "KDE"]:
        if acronym.startswith(prefix) and len(acronym) > len(prefix):
            suffix = acronym[len(prefix) :]
            # Handle special cases for suffixes
            match suffix:
                case "TANB":
                    suffix = "TAN"
                case "BNANB":
                    suffix = "BAN"
                case "KDB":
                    suffix = "$k$DB"
                case _:
                    suffix = suffix
            acronym = prefix + "-" + suffix
            break
    return acronym


# Convert arcs from IDs to names
def convert_arcs_to_names(bn: gum.BayesNet) -> list[tuple[str, str]]:
    """Convert arcs from node IDs to variable names"""
    arcs_with_names = []
    for source_id, target_id in bn.arcs():  # type: ignore library
        source_name = bn.variable(source_id).name()
        target_name = bn.variable(target_id).name()
        arcs_with_names.append((source_name, target_name))
    return arcs_with_names


# region Structural Distance
def node_presence_distance(nodes1: list, nodes2: list) -> int:
    """Calculates the Hamming distance between two lists of nodes

    Args:
        nodes1 (list): list of nodes
        nodes2 (list): list of nodes

    Returns:
        int: Number of node differences between the two lists
    """
    # Function to check the length of the symmetric difference in the nodes
    difference_nodes = set(nodes1).symmetric_difference(set(nodes2))
    return len(difference_nodes)


def node_type_distance(
    bn1: BayesianNetwork, bn2: BayesianNetwork, shared_nodes_list: list = []
) -> int:
    """Calculates the number of differences in the type of nodes between two Bayesian Networks and their shared nodes

    Args:
        bn1 (BayesianNetwork): Bayesian Network
        bn2 (BayesianNetwork): Bayesian Network
        shared_nodes_list (list): list of nodes shared between the two networks

    Returns:
        int: Number of differences in the type of nodes
    """
    if shared_nodes_list == []:
        shared_nodes_list = list(set(bn1.nodes()).intersection(set(bn2.nodes())))
    node_types1 = bn1.node_types()
    node_types2 = bn2.node_types()
    node_type_diff = 0

    for node in shared_nodes_list:
        if node_types1[node] != node_types2[node]:
            node_type_diff += 1
    return node_type_diff


def parametric_node_type_ratio(bn: BayesianNetwork) -> float:
    """
    Calculates the ratio of parametric (Linear Gaussian) nodes to the total number of
    parametric (Linear Gaussian) and CKDE nodes in a Bayesian network.
    The function supports both flat and nested (dict of dicts) node type structures.
    Args:
        bn (BayesianNetwork): The Bayesian network object, expected to have a `node_types()` method
            that returns a dictionary mapping node names to their types.
    Returns:
        float: The ratio of parametric (Linear Gaussian) nodes to the sum of parametric and CKDE nodes.
    Raises:
        ValueError: If the Bayesian network contains no parametric or CKDE nodes.
    """

    # Flatten node_types in case it's a dict of dicts or a flat dict
    node_types = bn.node_types()
    if all(isinstance(v, dict) for v in node_types.values()):
        # dict of dicts: collect all inner values
        node_types_list = [
            inner_v for v in node_types.values() for inner_v in v.values()
        ]
    else:
        # flat dict
        node_types_list = list(node_types.values())

    gaussian_node_count = node_types_list.count(
        pbn.LinearGaussianCPDType()
    )  # + node_types_list.count(pbn.DiscreteFactorType())
    ckde_node_count = node_types_list.count(pbn.CKDEType())

    total_node_count = gaussian_node_count + ckde_node_count
    if total_node_count == 0:
        raise ValueError("The Bayesian network has no nodes.")
    else:
        return gaussian_node_count / total_node_count


def hamming_distance(bn1: BayesianNetwork, bn2: BayesianNetwork) -> int:
    """
    Calculate the Hamming distance between two Bayesian networks.

    The Hamming distance is defined as the number of differing edges
    (arcs) between the skeletons of the two Bayesian networks. The
    skeleton of a Bayesian network is the set of edges without
    considering their direction.

    Args:
        bn1 (BayesianNetwork): The first Bayesian network.
        bn2 (BayesianNetwork): The second Bayesian network.

    Returns:
        int: The Hamming distance between the two Bayesian networks.

    Raises:
        AssertionError: If the sets of nodes in the two Bayesian networks are not identical.
    """
    assert set(bn1.nodes()) == set(bn2.nodes())
    arcs1 = bn1.arcs()
    arcs2 = bn2.arcs()
    arcs1_skeleton = set([tuple(sorted(arc)) for arc in arcs1])
    arcs2_skeleton = set([tuple(sorted(arc)) for arc in arcs2])
    difference_skeleton = arcs1_skeleton.symmetric_difference(arcs2_skeleton)
    return len(difference_skeleton)


def structural_hamming_distance(bn1: BayesianNetwork, bn2: BayesianNetwork) -> int:
    """
    Calculate the Structural Hamming Distance (SHD) between two Bayesian Networks.

    The SHD is defined as the number of edge insertions, deletions, or reversals needed to transform one network into the other.

    Args:
        bn1 (BayesianNetwork): The first Bayesian Network.
        bn2 (BayesianNetwork): The second Bayesian Network.

    Returns:
        int: The Structural Hamming Distance between the two Bayesian Networks.

    Raises:
        AssertionError: If the set of nodes in bn1 and bn2 are not identical.
    """
    assert set(bn1.nodes()) == set(bn2.nodes())
    arcs1 = set(bn1.arcs())
    arcs2 = set(bn2.arcs())
    shd_value = 0
    for est_arc in arcs1:
        if est_arc not in bn2.arcs():
            shd_value += 1
            s, d = est_arc
            if (d, s) in arcs2:
                arcs2.remove((d, s))

    for true_arc in arcs2:
        if true_arc not in arcs1:
            shd_value += 1

    return shd_value


# endregion


# region Parametric Distance
def gaussian_kullback_leibler_divergence(
    to: tuple[np.ndarray, np.ndarray], fr: tuple[np.ndarray, np.ndarray]
) -> float:
    """
    Computes the Kullback-Leibler (KL) divergence between two multivariate Gaussian distributions.
    The KL divergence is calculated from distribution `fr` (mean `m_fr`, covariance `S_fr`)
    to distribution `to` (mean `m_to`, covariance `S_to`):
        KL(N_fr || N_to) = 0.5 * [ tr(S_to^{-1} S_fr) + log(det(S_to)/det(S_fr)) + (m_to - m_fr)^T S_to^{-1} (m_to - m_fr) - k ]
    where k is the dimensionality of the mean vectors.
    Args:
        to (tuple[np.ndarray, np.ndarray]): A tuple containing the mean vector and covariance matrix of the target Gaussian (m_to, S_to).
        fr (tuple[np.ndarray, np.ndarray]): A tuple containing the mean vector and covariance matrix of the source Gaussian (m_fr, S_fr).
    Returns:
        float: The KL divergence value between the two Gaussian distributions.
    References:
        - https://gist.github.com/ChuaCheowHuan/18977a3e77c0655d945e8af60633e4df
        - https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html
    """

    m_to, S_to = to
    m_fr, S_fr = fr

    d = m_fr - m_to

    c, lower = scipy.linalg.cho_factor(S_fr)

    def solve(B):
        return scipy.linalg.cho_solve((c, lower), B)

    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    return ((term1 + term2 + term3 - len(d)) / 2.0).iat[0, 0]


def gaussian_jensen_shannon_divergence(
    p: tuple[np.ndarray, np.ndarray], q: tuple[np.ndarray, np.ndarray]
) -> float:
    """
    Computes the Jensen-Shannon divergence between two multivariate Gaussian distributions.

    The Jensen-Shannon divergence is a symmetric and smoothed version of the Kullback-Leibler divergence,
    measuring the similarity between two probability distributions. For Gaussian distributions, it is
    calculated using the means and covariance matrices of the distributions.

    Args:
        p (tuple[np.ndarray, np.ndarray]): A tuple containing the mean vector and covariance matrix of the first Gaussian distribution.
        q (tuple[np.ndarray, np.ndarray]): A tuple containing the mean vector and covariance matrix of the second Gaussian distribution.

    Returns:
        float: The Jensen-Shannon divergence between the two Gaussian distributions.
    """
    m_p, S_p = p
    m_q, S_q = q
    m = 0.5 * (m_p + m_q)
    S = 0.5 * (S_p + S_q)
    return 0.5 * gaussian_kullback_leibler_divergence(
        (m_p, S_p), (m, S)
    ) + 0.5 * gaussian_kullback_leibler_divergence((m_q, S_q), (m, S))


# endregion
