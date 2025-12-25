import os
from io import BytesIO
from PIL import Image
from skincare_ai.utils.config import settings


class BasicImageUtils:

    @classmethod
    async def read_image_file(cls, file, filename, cache=False) -> Image.Image:
        image = Image.open(BytesIO(file))
        if cache:
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            image.save(cache_dir / filename)
        return image
