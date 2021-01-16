from sklearn.decomposition import PCA


class PCAFeatures:
    """Transform feature vectors into self.dim-dimensional vectors"""

    def __init__(self, dim):
        self.dim = dim
        self.pca = PCA(n_components=dim)

    def fit(self, features):
        self.pca.fit(features)

    def transform(self, features):
        return self.pca.transform(features)
