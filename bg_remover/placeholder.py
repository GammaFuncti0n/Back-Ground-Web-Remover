from PIL import Image, ImageDraw

class Placeholder:
    def __init__(self):
        self.border_fraction = 0.12
    
    def remove(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        border = max(1, int(min(width, height) * self.border_fraction))

        result = rgb.copy()
        draw = ImageDraw.Draw(result)
        # Top, bottom, left, right white strips
        draw.rectangle([0, 0, width, border], fill="white")
        draw.rectangle([0, height - border, width, height], fill="white")
        draw.rectangle([0, 0, border, height], fill="white")
        draw.rectangle([width - border, 0, width, height], fill="white")
        return result