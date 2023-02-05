from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, \
    AgglomerativeClustering, DBSCAN, OPTICS, Birch

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from catboost import CatBoostClassifier


CLUSTERING_ALGORITHMS = {
    "KMeans": KMeans,
    "AffinityPropagation": AffinityPropagation,
    "MeanShift": MeanShift,
    "SpectralClustering": SpectralClustering,
    "AgglomerativeClustering": AgglomerativeClustering,
    "DBSCAN": DBSCAN,
    "OPTICS": OPTICS,
    "Birch": Birch
}

TREE_ALGORITHMS = ["CatBoost", "RandomForest"]
CLASSIFICATION_ALGORITHMS = {
    "SGD": SGDClassifier,
    "RandomForest": RandomForestClassifier,
    "CatBoost": CatBoostClassifier
}
