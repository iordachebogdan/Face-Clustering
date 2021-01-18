import face_recognition
import numpy as np
from tqdm import tqdm


class DlibFacenetFeatures:
    def __init__(self):
        pass

    def get_embedding(self, img):
        # face is whole image
        box = [(0, img.shape[1], img.shape[0], 0)]
        embedding = face_recognition.face_encodings(img, box)[0]
        return embedding

    def get_embeddings(self, imgs):
        embs = []
        for img in tqdm(imgs):
            embs.append(self.get_embedding(img))
        return np.array(embs)
