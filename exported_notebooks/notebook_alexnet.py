# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic("reset", "")


# %%
import torch
import torchvision.models as models
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from lib.alexnet import AlexNetFeatures
import lib.utils as utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


# %%
model = models.alexnet(pretrained=True).to(DEVICE)


# %%
model._modules


# %%
layer = model._modules.get("features")[7]
layer


# %%
train_imgs, train_labels = utils.load_train_images(as_pil=True)
print(len(train_imgs))


# %%
alexnet_features = AlexNetFeatures()


# %%
emb = alexnet_features.get_embedding(train_imgs[0])
emb.shape


# %%
emb


# %%
train_embs = alexnet_features.get_embeddings(train_imgs)
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
plt.title("AlexNet 2D clusters")
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
plt.title("AlexNet 3D clusters")
plt.show()
