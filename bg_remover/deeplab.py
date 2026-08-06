"""DeepLabV3 background removal (VainF / Pascal VOC)."""

from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from .network.modeling import deeplabv3_mobilenet, deeplabv3plus_resnet101


TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class DeepLab:
    def __init__(self, name: str):
        self.name = name

    @lru_cache(maxsize=1)
    def _init_model(self):
        if self.name == 'deeplabv3_mobilenet':
            checkpoint_path = Path(__file__).parent / "weights" / "deeplabv3_mobilenet.pth"
            checkpoint_url = "https://www.dropbox.com/s/uhksxwfcim3nkpo/best_deeplabv3_mobilenet_voc_os16.pth?dl=1"
        elif self.name == 'deeplabv3plus_resnet101':
            checkpoint_path = Path(__file__).parent / "weights" / "deeplabv3plus_resnet101.pth"
            checkpoint_url = "https://www.dropbox.com/s/bm3hxe7wmakaqc5/best_deeplabv3plus_resnet101_voc_os16.pth?dl=1"
        else:
            raise ValueError(f"Unknown processor: {self.name}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint_path.exists():
            print(f"Downloading {checkpoint_path.name} ...")
            urlretrieve(checkpoint_url, checkpoint_path)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if self.name == 'deeplabv3_mobilenet':
            self.model = deeplabv3_mobilenet(num_classes=21, output_stride=16, pretrained_backbone=False)
        elif self.name == 'deeplabv3plus_resnet101':
            self.model = deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=False)
        else:
            self.model = None
            raise ValueError(f"Unknown processor: {self.name}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
    
    
    def remove(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        resized, orig_size = _resize_for_infer(rgb)
        x = TRANSFORM(resized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x).argmax(1)[0].cpu().numpy()
        mask = Image.fromarray(((pred != 0) * 255).astype(np.uint8), mode="L")
        if mask.size != orig_size:
            mask = mask.resize(orig_size, Image.NEAREST)
        out = rgb.convert("RGBA")
        out.putalpha(mask)
        return out


def _resize_for_infer(image: Image.Image, size: int = 513):
    w, h = image.size
    scale = size / max(w, h)
    if scale >= 1.0:
        return image, (w, h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.BILINEAR), (w, h)