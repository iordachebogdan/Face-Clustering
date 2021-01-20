# %%
import glob
import pandas as pd
import os
from tqdm import tqdm
from collections import Counter
import random
import PIL
import matplotlib.pyplot as plt
import numpy as np


# %%
# read all identities' frame descriptions
all_identities_files = glob.glob("./data/frame_images_DB/*.txt")
print(f"{len(all_identities_files)} identities")
print(all_identities_files[:3])


# %%
identities_dfs = {}
for identity_file in tqdm(all_identities_files):
    name = os.path.basename(identity_file).split(".")[0]
    identities_dfs[name] = pd.read_csv(
        identity_file,
        header=None,
        names=[
            "filename",
            "ignore1",
            "x",
            "y",
            "width",
            "height",
            "ignore2",
            "ignore3",
        ],
    )
    identities_dfs[name].drop(columns=["ignore1", "ignore2", "ignore3"], inplace=True)
    identities_dfs[name].reset_index(inplace=True)


# %%
# find people with most frames
cnt = Counter()
for k, v in identities_dfs.items():
    cnt[k] = len(v)
print(cnt.most_common(20))


# %%
selected_people = [
    "Gabi_Zimmer",  # F
    "Linda_Franklin",  # F
    "Frank_Abagnale_Jr",  # M
    "Gary_Barnett",  # M
    "Rocco_Buttiglione",  # M
    "Natasha_McElhone",  # F
    "Peter_Goldmark",  # M
    "Darrell_Issa",  # M
    "Kristen_Breitweiser",  # F
    "Tessa_Jowell",  # F
]


# %%
example_file = identities_dfs[selected_people[1]]["filename"][0]
example_file = example_file.replace("\\", "/")
print(example_file)
x = identities_dfs[selected_people[1]]["x"][0]
y = identities_dfs[selected_people[1]]["y"][0]
width = identities_dfs[selected_people[1]]["width"][0]
height = identities_dfs[selected_people[1]]["height"][0]

x_min = x - width // 2
y_min = y - height // 2

example_img = np.array(PIL.Image.open("data/frame_images_DB/" + example_file))
print(example_img.shape)
example_img = example_img[y_min : y_min + height, x_min : x_min + width]
print(example_img.shape)
plt.imshow(example_img)


# %%
NUM_FRAMES = 500
TRAIN_PCT = 0.7

os.mkdir("dataset/train")
os.mkdir("dataset/test")
for i, name in enumerate(tqdm(selected_people)):
    df_selected_frames = identities_dfs[name].sample(n=NUM_FRAMES)
    os.mkdir(f"dataset/train/{i}")
    os.mkdir(f"dataset/test/{i}")
    for j, (filename, x, y, width, height) in enumerate(
        zip(
            df_selected_frames["filename"],
            df_selected_frames["x"],
            df_selected_frames["y"],
            df_selected_frames["width"],
            df_selected_frames["height"],
        )
    ):
        filename = filename.replace("\\", "/")
        x_min = x - width // 2
        y_min = y - height // 2

        img = np.array(PIL.Image.open("data/frame_images_DB/" + filename))
        img = img[y_min : y_min + height, x_min : x_min + width]
        img = PIL.Image.fromarray(img)
        img.save(
            f"dataset/{'train' if j < NUM_FRAMES * TRAIN_PCT else 'test'}/{i}/{j}.jpg"
        )


# %%
fig, ax = plt.subplots(3, 10)
fig.subplots_adjust(bottom=0.5)
for i in range(len(selected_people)):
    for j in range(3):
        img = np.array(PIL.Image.open(f"dataset/train/{i}/{j}.jpg"))
        ax[j, i].imshow(img)
        ax[j, i].axis("off")
plt.show()
