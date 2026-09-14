from sklearn.datasets import load_breast_cancer, load_iris, load_wine

SKLEARN_DATASET_NAME_DICT = {
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
    "breast_cancer": load_breast_cancer,
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
    "iris": load_iris,
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine
    "wine": load_wine,
    # # NOTE: This datasets have too much dimensionality.
    # # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html#sklearn.datasets.fetch_covtype
    # "covertype": fetch_covtype(as_frame=True, random_state=42),
    # # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
    # "digits": load_digits(n_class=10, as_frame=True),
    # # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html#sklearn.datasets.fetch_kddcup99
    # "kddcup99": fetch_kddcup99(
    #     subset=None, percent10=True, as_frame=True, random_state=42
    # ),
}
UCI_DATASET_NAME_DICT = {
    "algerian_forest_fires": 547,  # https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset
    "banknote_authentication": 267,  # https://archive.ics.uci.edu/dataset/267/banknote+authentication
    "blood_transfusion": 176,  # https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center
    "breast_cancer_coimbra": 451,  # https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra
    "breast_cancer_wisconsin": 15,  # https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
    "electrical_grid_stability": 471,  # https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data
    "glass": 42,  # https://archive.ics.uci.edu/dataset/42/glass+identification
    "haberman's_survival": 43,  # https://archive.ics.uci.edu/dataset/43/haberman+s+survival
    "HTRU2": 372,  # https://archive.ics.uci.edu/dataset/372/htru2
    "magic_gamma_telescope": 159,  # https://ucimlrepo.github.io/ucimlrepo-docs/datasets/magic_gamma_telescope/
    "mammographic_mass": 161,  # https://archive.ics.uci.edu/dataset/161/mammographic+mass
    "occupancy_detection": 357,  # https://archive.ics.uci.edu/dataset/357/occupancy+detection
    "page_blocks_classification": 78,  # https://archive.ics.uci.edu/dataset/78/page+blocks+classification
    "parkinson's": 174,  # https://archive.ics.uci.edu/dataset/174/parkinsons
    "rice": 545,  # https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
    "user_knowledge_modeling": 257,  # https://archive.ics.uci.edu/dataset/257/user+knowledge+modeling
    "vertebral_column": 212,  # https://archive.ics.uci.edu/dataset/212/vertebral+column
    "waveform_generator": 107,  # https://archive.ics.uci.edu/dataset/107/waveform+database+generator+version+1
    "website_phishing": 379,  # https://archive.ics.uci.edu/dataset/379/website+phishing
    "wholesale_customers": 292,  # https://archive.ics.uci.edu/dataset/292/wholesale+customers
    # NOTE: NaN log-likelihood for KDE models, we fill them with NAN_LOGL_VALUE
    "ecoli": 39,  # https://archive.ics.uci.edu/dataset/39/ecoli
    "pen_digits": 81,  # https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits
    # "yeast": 110,  # https://archive.ics.uci.edu/dataset/110/yeast
}

PROBLEMATIC_UCI_DATASET_DICT = {
    # NOTE: Cannot split n instances into 10 folds with n<10 in multinet learning (No results for these)
    "liver_disorders": 60,  #  https://archive.ics.uci.edu/dataset/60/liver+disorders
    "statlog_shuttle": 148,  # https://archive.ics.uci.edu/dataset/148/statlog+shuttle
    # NOTE: Not enough instances per class (value_counts < min_class_sample_size)
    "image_segmentation": 50,  # https://archive.ics.uci.edu/dataset/50/image+segmentation
    # NOTE: 1 class only, so we cannot split it into train and test sets
    "breast_cancer_wisconsin_prognostic": 16,  # https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic
    "cervical_cancer_behavior_risk": 537,  # https://archive.ics.uci.edu/dataset/537/cervical+cancer+behavior+risk
}

DATASET_NAME_LIST = sorted(
    SKLEARN_DATASET_NAME_DICT.keys() | UCI_DATASET_NAME_DICT.keys()
)
