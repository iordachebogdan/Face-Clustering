# %%
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA

from lib.dlib_facenet import DlibFacenetFeatures
import lib.utils as utils


# %%
train_imgs, train_labels = utils.load_train_images_cv_grayscale(grayscale=False)
print(len(train_imgs))


# %%
plt.imshow(train_imgs[0])


# %%
dlib_facenet_feature_manager = DlibFacenetFeatures()


# %%
emb = dlib_facenet_feature_manager.get_embedding(train_imgs[0])
print(emb.shape)
emb


# %%
train_embs = dlib_facenet_feature_manager.get_embeddings(train_imgs)
train_embs.shape


# %%
pca_2 = PCA(n_components=2)
pca_2.fit(train_embs)
train_features_2d = pca_2.transform(train_embs)

plt.scatter(
    train_features_2d[:, 0],
    train_features_2d[:, 1],
    s=3,
    c=train_labels,
    cmap="viridis",
)
plt.title("Dlib Facenet 2D clusters")
plt.show()


# %%
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")

pca_3 = PCA(n_components=3)
pca_3.fit(train_embs)
train_features_3d = pca_3.transform(train_embs)

ax.scatter(
    train_features_3d[:, 0],
    train_features_3d[:, 1],
    train_features_3d[:, 2],
    s=3,
    c=train_labels,
    cmap="viridis",
)
plt.title("Dlib Facenet 3D clusters")
plt.show()


# %%
test_imgs, test_labels = utils.load_test_images_cv_grayscale(grayscale=False)
test_embs = dlib_facenet_feature_manager.get_embeddings(test_imgs)
test_embs.shape


# %%
with open("./.emb_cache/dlib/train_embs.bin", "wb") as f:
    pickle.dump(train_embs, f)
with open("./.emb_cache/dlib/train_embs.bin", "rb") as f:
    saved_train_embs = pickle.load(f)
    print((train_embs == saved_train_embs).mean())


# %%
with open("./.emb_cache/dlib/test_embs.bin", "wb") as f:
    pickle.dump(test_embs, f)
with open("./.emb_cache/dlib/test_embs.bin", "rb") as f:
    saved_test_embs = pickle.load(f)
    print((test_embs == saved_test_embs).mean())
