"""IS-Net (DIS) background removal — https://github.com/xuebinqin/DIS"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize

from .isnet_model import ISNetDIS

# Official general-use weights (HF mirror of Google Drive from DIS README).
# Official Drive: https://drive.google.com/file/d/1klUkUnQFAxPdFu9XE3Vhnn-e72CVz6oM
CHECKPOINT_URL = (
    "https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth"
)
CHECKPOINT_PATH = Path(__file__).parent / "weights" / "isnet-general-use.pth"
INPUT_SIZE = (1024, 1024)


class ISNet:
    def __init__(self, name: str = "isnet"):
        self.name = name

    def _init_model(self):
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not CHECKPOINT_PATH.exists():
            print(f"Downloading {CHECKPOINT_PATH.name} from\n  {CHECKPOINT_URL}")
            urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = ISNetDIS()
        state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def remove(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        h, w = rgb.size[1], rgb.size[0]

        # Same preprocess as DIS/IS-Net/Inference.py
        im = torch.tensor(np.array(rgb), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        im = F.interpolate(im, size=INPUT_SIZE, mode="bilinear", align_corners=False)
        x = torch.divide(im, 255.0)
        x = normalize(x, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(self.device)

        with torch.no_grad():
            result = self.model(x)
            # result[0][0] — first side output
            mask = F.interpolate(result[0][0], size=(h, w), mode="bilinear", align_corners=False)
            mask = mask.squeeze()
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        mask_img = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8), mode="L")
        out = rgb.convert("RGBA")
        out.putalpha(mask_img)
        return out
