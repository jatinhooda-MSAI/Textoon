import imagehash
from PIL import Image
from pathlib import Path

IMG_DIR = Path("raw_images")

def phash_file(filename):
    """Compute pHash for one file. Returns (filename, phash_string) or (filename, None)."""
    try:
        img = Image.open(IMG_DIR / filename)
        return filename, str(imagehash.phash(img))
    except Exception:
        return filename, None