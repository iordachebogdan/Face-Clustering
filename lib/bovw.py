import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler


def compute_keypoints_and_descriptors(sift, img):
    key_points, descriptors = sift.detectAndCompute(img, None)
    return key_points, descriptors


def plot_keypoints(key_points, img):
    res = img.copy()
    res = cv2.drawKeypoints(
        img, key_points, res, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    plt.imshow(res)


class BOVWFeatures:
    """Transform cv2 images into BOVW histograms with `dim` vocabulary size"""

    def __init__(self, dim):
        self.dim = dim
        self.sift = cv2.SIFT_create()
        self.vocab = None
        self.scaler = StandardScaler()

    def fit_and_get_histograms(self, imgs):
        # get all descriptors
        all_descriptors = []
        for i, img in enumerate(imgs):
            _, descriptors = compute_keypoints_and_descriptors(self.sift, img)
            all_descriptors.append(descriptors)

        stacked_descriptors = all_descriptors[0]
        for descriptors in all_descriptors[1:]:
            if descriptors is None:
                continue
            stacked_descriptors = np.vstack((stacked_descriptors, descriptors))
        stacked_descriptors = stacked_descriptors.astype(np.float64)

        # perform K-Means on descriptors
        print("Performing K-Means")
        self.vocab, variance = kmeans(stacked_descriptors, self.dim, 1)

        # create histograms
        print("Computing histograms")
        features = np.zeros((len(imgs), self.dim), dtype=np.float64)
        for i, descriptors in enumerate(all_descriptors):
            if descriptors is None:
                continue
            words, distance = vq(descriptors, self.vocab)
            for word in words:
                features[i, word] += 1

        # fit standardizer
        self.scaler.fit(features)
        features = self.scaler.transform(features)

        return features

    def get_histograms(self, imgs):
        # get all descriptors
        all_descriptors = []
        for img in imgs:
            _, descriptors = compute_keypoints_and_descriptors(self.sift, img)
            all_descriptors.append(descriptors)

        # compute features
        features = np.zeros((len(imgs), self.dim), dtype=np.float64)
        for i, descriptors in enumerate(all_descriptors):
            if descriptors is None:
                continue
            words, distance = vq(descriptors, self.vocab)
            for word in words:
                features[i, word] += 1

        # apply standardization
        features = self.scaler.transform(features)

        return features
