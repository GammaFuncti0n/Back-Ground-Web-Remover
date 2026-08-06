import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForImageSegmentation
from PIL import Image
from torchvision import transforms as T

HF_REPO = "ZhengPeng7/BiRefNet"
HF_FILE = Path(__file__).parent / "weights"

IMAGE_SIZE = (1024, 1024)
TRANSFORM = T.Compose(
    [
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class BiRefNet:
    def __init__(self, name: str = "birefnet"):
        self.name = name
    
    def _init_model(self):
        self.model = AutoModelForImageSegmentation.from_pretrained(HF_REPO, trust_remote_code=True, cache_dir=HF_FILE)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device).eval().half()

    def remove(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        x = TRANSFORM(rgb).unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            preds = self.model(x)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = T.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        return image