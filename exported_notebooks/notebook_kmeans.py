# %% [markdown]
# # K Means

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from munkres import Munkres

from lib.pca import PCAFeatures
import lib.bovw as bovw
from lib.alexnet import AlexNetFeatures
import lib.utils as utils

# %% [markdown]
# ## 1. PCA Features

# %%
train_imgs, train_labels = utils.load_train_images()
print(len(train_imgs))


# %%
train_resized_imgs = utils.resize_imgs(train_imgs)
mean_img = utils.get_mean_img(train_resized_imgs)
train_norm_imgs = utils.basic_imgs_normalization(train_resized_imgs)
train_features = utils.image_liniarization(train_norm_imgs)
train_features.shape


# %%
pca_300 = PCAFeatures(dim=300)
pca_300.fit(train_features)
train_features_reduced_300 = pca_300.transform(train_features)
print(train_features_reduced_300.shape)


# %%
def kmeans_clustering(n_clusters, n_true_clusters, train_features, train_labels):
    """K Means clustering using given features, returns a fitted estimator
    and the predicted labels for the training data
    """
    # Fit the estimator and predict the labels
    k_means = KMeans(n_clusters, random_state=42)
    k_means.fit(train_features)
    predicted_labels = k_means.predict(train_features)

    # compute silhouette scores for every sample
    sil_avg = metrics.silhouette_score(train_features, predicted_labels)
    sample_sil_values = metrics.silhouette_samples(train_features, predicted_labels)

    # the silhouette plot for each cluster
    fig, ax = plt.subplots()
    ax.set_xlim([-0.1, 1])
    skip_between = 10
    ax.set_ylim([0, len(train_features) + (n_clusters + 1) * skip_between])

    y_min = 10
    for i in range(n_clusters):
        sil_values = sample_sil_values[predicted_labels == i]
        sil_values.sort()

        size = sil_values.shape[0]
        y_max = y_min + size

        color = cm.nipy_spectral(i / n_clusters)
        ax.fill_betweenx(
            np.arange(y_min, y_max),
            0,
            sil_values,
            facecolor=color,
            edgecolor=color,
        )

        ax.text(-0.1, y_min + 0.5 * size, str(i))
        y_min = y_max + skip_between

    ax.set_title(f"Silhouette plot for {n_clusters} clusters")
    ax.set_xlabel("Silhouette scores")
    ax.set_ylabel("Cluster")

    # plot the silhouette mean score
    ax.axvline(x=sil_avg, color="red")

    ax.set_yticks([])
    ax.set_xticks(np.arange(6) / 5)

    plt.show()

    # plot for K Means clustering vs. actual distribution
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # use PCA for 2D plot
    pca_2 = PCAFeatures(2)
    pca_2.fit(train_features)
    train_features_2 = pca_2.transform(train_features)

    colors = cm.nipy_spectral(predicted_labels.astype(float) / n_clusters)
    ax1.scatter(
        train_features_2[:, 0], train_features_2[:, 1], marker=".", s=20, c=colors
    )
    # also plot cluster centers
    centers = pca_2.transform(k_means.cluster_centers_)
    ax1.scatter(
        centers[:, 0], centers[:, 1], marker="o", c="white", s=100, edgecolor="k"
    )
    ax1.set_title("Clustered data")
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)

    colors = cm.nipy_spectral(np.array(train_labels).astype(float) / n_true_clusters)
    ax2.scatter(
        train_features_2[:, 0], train_features_2[:, 1], marker=".", s=20, c=colors
    )
    ax2.set_title("Actual distribution of the data")
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)

    plt.show()

    return k_means, predicted_labels


# %%
k_means, predicted_labels = kmeans_clustering(
    10, 10, train_features_reduced_300, train_labels
)
predicted_labels.shape

# %% [markdown]
# Use the Hungarian algorithm in order to find the best matching between the clusters' labels and the actual class labels.

# %%
matching = Munkres()
contingency_matrix = metrics.cluster.contingency_matrix(train_labels, predicted_labels)
best_match = {}
for true, pred in matching.compute(contingency_matrix.max() - contingency_matrix):
    best_match[pred] = true
print(best_match)


# %%
remapped_predicted_labels = [best_match[label] for label in predicted_labels]


# %%
print(f"Train ACC: {metrics.accuracy_score(train_labels, remapped_predicted_labels)}")


# %%
print(
    f"Train NMI: {metrics.normalized_mutual_info_score(train_labels, remapped_predicted_labels)}"
)


# %%
test_imgs, test_labels = utils.load_test_images()
print(len(test_imgs))


# %%
test_resized_imgs = utils.resize_imgs(test_imgs)
test_norm_imgs = utils.basic_imgs_normalization(test_resized_imgs, mean_img=mean_img)
test_features = utils.image_liniarization(test_norm_imgs)
test_features.shape


# %%
test_features_reduced_300 = pca_300.transform(test_features)
print(test_features_reduced_300.shape)


# %%
test_predicted_labels = k_means.predict(test_features_reduced_300)
remapped_test_predicted_labels = [best_match[label] for label in test_predicted_labels]
print(len(remapped_test_predicted_labels))


# %%
print(
    f"Test ACC: {metrics.accuracy_score(test_labels, remapped_test_predicted_labels)}"
)


# %%
print(
    f"Test NMI: {metrics.normalized_mutual_info_score(test_labels, remapped_test_predicted_labels)}"
)

# %% [markdown]
# ## 2. BOVW

# %%
train_imgs, train_labels = utils.load_train_images_cv_grayscale()
print(len(train_imgs))


# %%
bag_of_visual_words = bovw.BOVWFeatures(dim=4000)
train_features = bag_of_visual_words.fit_and_get_histograms(train_imgs)
print(train_features.shape)


# %%
train_features_norm = train_features / np.linalg.norm(train_features, axis=1).reshape(
    (len(train_features), 1)
)
train_features_norm.shape


# %%
k_means, predicted_labels = kmeans_clustering(10, 10, train_features_norm, train_labels)
predicted_labels.shape


# %%
matching = Munkres()
contingency_matrix = metrics.cluster.contingency_matrix(train_labels, predicted_labels)
best_match = {}
for true, pred in matching.compute(contingency_matrix.max() - contingency_matrix):
    best_match[pred] = true
print(best_match)


# %%
remapped_predicted_labels = [best_match[label] for label in predicted_labels]


# %%
print(f"Train ACC: {metrics.accuracy_score(train_labels, remapped_predicted_labels)}")


# %%
print(
    f"Train NMI: {metrics.normalized_mutual_info_score(train_labels, remapped_predicted_labels)}"
)


# %%
test_imgs, test_labels = utils.load_test_images_cv_grayscale()
print(len(test_imgs))


# %%
test_features = bag_of_visual_words.get_histograms(test_imgs)
test_features_norm = test_features / np.linalg.norm(test_features, axis=1).reshape(
    (len(test_features), 1)
)
test_predicted_labels = k_means.predict(test_features_norm)
remapped_test_predicted_labels = [best_match[label] for label in test_predicted_labels]
print(len(remapped_test_predicted_labels))


# %%
print(
    f"Test ACC: {metrics.accuracy_score(test_labels, remapped_test_predicted_labels)}"
)


# %%
print(
    f"Test NMI: {metrics.normalized_mutual_info_score(test_labels, remapped_test_predicted_labels)}"
)

# %% [markdown]
# ## 3. AlexNet Features

# %%
train_imgs, train_labels = utils.load_train_images(as_pil=True)
print(len(train_imgs))


# %%
alexnet_features = AlexNetFeatures()
train_features = alexnet_features.get_embeddings(train_imgs)
train_features.shape


# %%
k_means, predicted_labels = kmeans_clustering(10, 10, train_features, train_labels)
predicted_labels.shape


# %%
matching = Munkres()
contingency_matrix = metrics.cluster.contingency_matrix(train_labels, predicted_labels)
best_match = {}
for true, pred in matching.compute(contingency_matrix.max() - contingency_matrix):
    best_match[pred] = true
print(best_match)


# %%
remapped_predicted_labels = [best_match[label] for label in predicted_labels]


# %%
print(f"Train ACC: {metrics.accuracy_score(train_labels, remapped_predicted_labels)}")


# %%
print(
    f"Train NMI: {metrics.normalized_mutual_info_score(train_labels, remapped_predicted_labels)}"
)


# %%
test_imgs, test_labels = utils.load_test_images(as_pil=True)
print(len(test_imgs))


# %%
test_features = alexnet_features.get_embeddings(test_imgs)
test_predicted_labels = k_means.predict(test_features)
remapped_test_predicted_labels = [best_match[label] for label in test_predicted_labels]
print(len(remapped_test_predicted_labels))


# %%
print(
    f"Test ACC: {metrics.accuracy_score(test_labels, remapped_test_predicted_labels)}"
)


# %%
print(
    f"Test NMI: {metrics.normalized_mutual_info_score(test_labels, remapped_test_predicted_labels)}"
)

# %% [markdown]
# ## 4. Dlib Facenet Features

# %%
train_features = None
with open("./.emb_cache/dlib/train_embs.bin", "rb") as f:
    train_features = pickle.load(f)
train_features.shape


# %%
k_means, predicted_labels = kmeans_clustering(10, 10, train_features, train_labels)
predicted_labels.shape


# %%
matching = Munkres()
contingency_matrix = metrics.cluster.contingency_matrix(train_labels, predicted_labels)
best_match = {}
for true, pred in matching.compute(contingency_matrix.max() - contingency_matrix):
    best_match[pred] = true
print(best_match)


# %%
remapped_predicted_labels = [best_match[label] for label in predicted_labels]


# %%
print(f"Train ACC: {metrics.accuracy_score(train_labels, remapped_predicted_labels)}")


# %%
print(
    f"Train NMI: {metrics.normalized_mutual_info_score(train_labels, remapped_predicted_labels)}"
)


# %%
test_features = None
with open("./.emb_cache/dlib/test_embs.bin", "rb") as f:
    test_features = pickle.load(f)
test_predicted_labels = k_means.predict(test_features)
remapped_test_predicted_labels = [best_match[label] for label in test_predicted_labels]
print(len(remapped_test_predicted_labels))


# %%
print(
    f"Test ACC: {metrics.accuracy_score(test_labels, remapped_test_predicted_labels)}"
)


# %%
print(
    f"Test NMI: {metrics.normalized_mutual_info_score(test_labels, remapped_test_predicted_labels)}"
)
