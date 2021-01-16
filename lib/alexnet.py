import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AlexNetFeatures:
    def __init__(self):
        self.model = models.alexnet(pretrained=True).to(DEVICE)
        self.layer = self.model._modules.get("features")[7]
        self.model.eval()

        self.scaler = transforms.Scale((224, 224))
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def get_embedding(self, img):
        img_tensor = torch.autograd.Variable(
            self.normalizer(self.to_tensor(self.scaler(img))).unsqueeze(0)
        ).to(DEVICE)

        # hook for saving the embedding after the specified layer
        embedding = None

        def save_result(m, i, output):
            nonlocal embedding
            embedding = torch.flatten(output.clone().detach()).cpu().numpy()

        # register the hook
        handle = self.layer.register_forward_hook(save_result)

        # pass image through model
        self.model(img_tensor)

        # remove hook
        handle.remove()

        return embedding

    def get_embeddings(self, imgs):
        embs = []
        for img in imgs:
            embs.append(self.get_embedding(img))
        return np.array(embs)
