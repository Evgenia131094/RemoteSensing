from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, \
    AgglomerativeClustering, DBSCAN, OPTICS, Birch

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
