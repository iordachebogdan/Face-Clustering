# %% [markdown]
# # SVM

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
import pickle

from lib.pca import PCAFeatures
import lib.bovw as bovw
from lib.alexnet import AlexNetFeatures
import lib.utils as utils

# %% [markdown]
# ## 1. PCA features

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
svm = LinearSVC(verbose=True)
svm.fit(train_features_reduced_300, train_labels)


# %%
predict_train = svm.predict(train_features_reduced_300)
print(f"Train ACC: {metrics.accuracy_score(train_labels, predict_train)}")


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
predict_test = svm.predict(test_features_reduced_300)
print(f"Test ACC: {metrics.accuracy_score(test_labels, predict_test)}")

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
svm = LinearSVC(verbose=True)
svm.fit(train_features, train_labels)


# %%
predict_train = svm.predict(train_features)
print(f"Train ACC: {metrics.accuracy_score(train_labels, predict_train)}")


# %%
test_imgs, test_labels = utils.load_test_images_cv_grayscale()
print(len(test_imgs))


# %%
test_features = bag_of_visual_words.get_histograms(test_imgs)
test_features.shape


# %%
predict_test = svm.predict(test_features)
print(f"Test ACC: {metrics.accuracy_score(test_labels, predict_test)}")

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
svm = LinearSVC(verbose=True)
svm.fit(train_features, train_labels)


# %%
predict_train = svm.predict(train_features)
print(f"Train ACC: {metrics.accuracy_score(train_labels, predict_train)}")


# %%
test_imgs, test_labels = utils.load_test_images(as_pil=True)
print(len(test_imgs))


# %%
test_features = alexnet_features.get_embeddings(test_imgs)
test_features.shape


# %%
predict_test = svm.predict(test_features)
print(f"Test ACC: {metrics.accuracy_score(test_labels, predict_test)}")

# %% [markdown]
# ## 4. Dlib Facenet Features

# %%
train_features = None
with open("./.emb_cache/dlib/train_embs.bin", "rb") as f:
    train_features = pickle.load(f)
train_features.shape


# %%
svm = LinearSVC(verbose=True)
svm.fit(train_features, train_labels)


# %%
predict_train = svm.predict(train_features)
print(f"Train ACC: {metrics.accuracy_score(train_labels, predict_train)}")


# %%
test_features = None
with open("./.emb_cache/dlib/test_embs.bin", "rb") as f:
    test_features = pickle.load(f)
test_features.shape


# %%
predict_test = svm.predict(test_features)
print(f"Test ACC: {metrics.accuracy_score(test_labels, predict_test)}")
