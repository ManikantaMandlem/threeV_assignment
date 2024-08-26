import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torch.nn.functional as TF
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"model_sc")))
from swin_transformer import SwinUNet


class InfObject:
    def __init__(
        self, device="cpu", m_size=(224, 224), conf_thre=0.5, model_weights=None
    ):
        self.m_width = m_size[0]
        self.m_height = m_size[1]
        self.orig_width = None
        self.orig_height = None
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.Normalize(mean=[0.5], std=[0.5])]
        )
        self.image = None
        self.mask = None
        self.threshold = conf_thre
        self.device = torch.device(device)
        self.orig_image_path = None
        self.pred_mask_path = None
        if model_weights:
            weights = torch.load(model_weights, map_location=self.device)
            model_version = weights["config"]["version"]
            print(f"Loading trained model: {model_version}")
            self.ckpt = weights["state_dict"]
        else:
            print("Initializing the segmentation model randomly")
            self.ckpt = None
        self.model = self._load_model()

    def _load_model(self):
        model = SwinUNet(
            H=self.m_width,
            W=self.m_height,
            ch=1,
            C=32,
            num_class=1,
            num_blocks=3,
            patch_size=4,
        ).to(self.device)
        if self.ckpt:
            model.load_state_dict(self.ckpt)
        print("Segmentation model initialized")
        return model

    def load_image(self, image_path):
        self.orig_image_path = image_path.replace(".tif", ".png")
        self.pred_mask_path = image_path.replace(".tif", "_mask.png")
        self.image = Image.open(image_path)
        self.image.save(self.orig_image_path)
        self.orig_width, self.orig_height = self.image.size
        # self.image.show()
        self.image = self.to_tensor(self.image)
        self.image = F.resize(
            self.image, (self.m_width, self.m_height), interpolation=Image.BILINEAR
        )
        self.image = self.normalize(self.image)
        print("Image loaded successfully and looks beautiful!")

    def _make_mask(self):
        self.mask = self.model(self.image.unsqueeze(0))
        self.mask = torch.sigmoid(self.mask)
        self.mask = 1 * (self.mask >= self.threshold).float()
        self.mask = TF.interpolate(
            self.mask, size=(self.orig_height, self.orig_width), mode="nearest"
        )
        self.mask = self.mask[0][0]

    def save_result(self):
        self._make_mask()
        display_mask = self.mask * 255
        display_mask = display_mask.cpu().byte().numpy()
        display_mask = Image.fromarray(display_mask)
        # display_mask.show()
        display_mask.save(self.pred_mask_path)
        print("Boom! Mask created successfully!")
