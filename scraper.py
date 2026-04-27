"""
Safebooru Image Scraper for Kid-Friendly Cartoon Character Generation
======================================================================
Downloads single-character images with attribute tags from Safebooru's API.
Optimized with 1000 posts/page API calls and multithreaded image downloads.

Usage:
    python scraper.py --num_images 10000 --output_dir ./raw_images --workers 8
"""

import os
import csv
import time
import argparse
import requests
import threading
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Tag Configuration ──────────────────────────────────────────────────────
# Kid-friendly categories: animals, magical/fantasy characters, 
# storybook-style attributes, colors, expressions, clothing.

ATTRIBUTE_TAGS = {
    # ── Character Type ─────────────────────────────────────────────────
    "character_type": [
        "animal_ears", "cat_ears", "dog_ears", "rabbit_ears", "fox_ears",
        "tail", "cat_tail", "dog_tail", "fox_tail",
        "wings", "angel_wings", "demon_wings", "fairy_wings", "butterfly_wings",
        "horns", "pointy_ears", "robot", "monster",
    ],
    # ── Animal / Creature ──────────────────────────────────────────────
    "animal": [
        "cat", "dog", "rabbit", "bear", "bird", "owl", "dragon",
        "unicorn", "fox", "deer", "penguin", "frog", "fish",
        "wolf", "lion", "mouse", "panda", "dinosaur",
    ],
    # ── Hair Color ─────────────────────────────────────────────────────
    "hair_color": [
        "blonde_hair", "brown_hair", "black_hair", "red_hair", "blue_hair",
        "green_hair", "pink_hair", "purple_hair", "white_hair", "orange_hair",
    ],
    # ── Hair Style ─────────────────────────────────────────────────────
    "hair_style": [
        "long_hair", "short_hair", "twintails", "ponytail", "braids",
        "bob_cut", "curly_hair",
    ],
    # ── Eye Color ──────────────────────────────────────────────────────
    "eye_color": [
        "blue_eyes", "red_eyes", "green_eyes", "brown_eyes", "purple_eyes",
        "yellow_eyes",
    ],
    # ── Expression ─────────────────────────────────────────────────────
    "expression": [
        "smile", "open_mouth", "closed_eyes", "surprised", "blush",
        "grin", "happy", "laughing", "winking", ":d",
    ],
    # ── Clothing / Outfit ──────────────────────────────────────────────
    "clothing": [
        "dress", "shirt", "jacket", "hoodie", "cape", "robe",
        "armor", "overalls", "apron", "crown", "tiara",
        "hat", "witch_hat", "wizard_hat", "beret", "helmet",
    ],
    # ── Accessories ────────────────────────────────────────────────────
    "accessories": [
        "glasses", "ribbon", "bow", "scarf", "gloves",
        "backpack", "bag", "wand", "staff", "sword", "shield",
        "book", "flower", "star_(symbol)", "heart",
    ],
    # ── Pose / Action ──────────────────────────────────────────────────
    "pose_action": [
        "standing", "sitting", "walking", "running", "jumping",
        "flying", "arms_up", "hand_on_hip", "crossed_arms",
        "waving", "hugging", "sleeping", "reading",
    ],
    # ── Setting / Background ───────────────────────────────────────────
    "setting": [
        "outdoors", "indoors", "forest", "sky", "castle",
        "garden", "underwater", "night_sky", "starry_sky", "cloud",
        "rainbow", "snow", "beach", "mountain",
    ],
    # ── Style / Mood ───────────────────────────────────────────────────
    "style": [
        "chibi", "cute", "colorful", "sparkle", "magical_girl",
        "fantasy", "fairy_tale", "storybook",
    ],
}

# Flatten all attribute tags into a single list (this becomes our attribute vector)
ALL_ATTRIBUTE_TAGS = []
for category, tags in ATTRIBUTE_TAGS.items():
    ALL_ATTRIBUTE_TAGS.extend(tags)

# Tags to EXCLUDE (not kid-friendly or not useful)
EXCLUDED_TAGS = [
    "comic", "multiple_girls", "multiple_boys", "monochrome",
    "realistic", "photo", "3d", "multiple_views", "blood",
    "horror", "gore", "weapon", "gun", "cigarette", "alcohol",
    "bikini", "swimsuit", "underwear", "lingerie", "nude",
    "suggestive", "cleavage", "midriff", "miniskirt",
    "highres",  # just a quality tag, not useful as attribute
]


# ── Safebooru API ──────────────────────────────────────────────────────────

SAFEBOORU_API = "https://safebooru.org/index.php"

def append_row(metadata_path, fieldnames, row):
    with open(metadata_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

def fetch_page(page: int, tags: str, limit: int = 1000) -> list:
    """
    Fetch a single page of results from Safebooru API.
    Safebooru supports up to 1000 posts per request.
    """
    params = {
        "page": "dapi",
        "s": "post",
        "q": "index",
        "json": 1,
        "tags": tags,
        "limit": limit,
        "pid": page,
    }
    try:
        resp = requests.get(SAFEBOORU_API, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return []
    except (requests.RequestException, ValueError) as e:
        print(f"  [WARN] Page {page} fetch failed: {e}")
        return []


def build_search_tags() -> str:
    """Build the search query string for Safebooru."""
    tags = ["solo"]
    for tag in EXCLUDED_TAGS:
        tags.append(f"-{tag}")
    return " ".join(tags)


def extract_attributes(tag_string: str) -> dict:
    """
    Given the full tag string from Safebooru, extract our attribute vector.
    Returns a dict of {tag: 1/0} for all tags in ALL_ATTRIBUTE_TAGS.
    """
    image_tags = set(tag_string.strip().split())
    attributes = {}
    for tag in ALL_ATTRIBUTE_TAGS:
        attributes[tag] = 1 if tag in image_tags else 0
    return attributes


def has_minimum_attributes(attributes: dict, min_attrs: int = 5) -> bool:
    """Check that an image has at least `min_attrs` known attributes."""
    return sum(attributes.values()) >= min_attrs


def download_single_image(post: dict, output_path: Path) -> dict | None:
    """
    Download and validate a single image. Returns metadata row or None.
    Thread-safe.
    """
    post_id = str(post.get("id", ""))
    directory = post.get("directory", "")
    image_name = post.get("image", "")
    tag_string = post.get("tags", "")
    
    if not directory or not image_name:
        return None
    
    image_url = f"https://safebooru.org/images/{directory}/{image_name}"
    save_filename = f"{post_id}.png"
    save_path = output_path / save_filename
    
    try:
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        
        # Quality filters
        w, h = img.size
        if w < 256 or h < 256:
            return None
        if max(w, h) / min(w, h) > 2.0:
            return None
        
        # Convert and save
        img = img.convert("RGB")
        img.save(save_path, "PNG")
        
        # Build row
        attributes = extract_attributes(tag_string)
        if not has_minimum_attributes(attributes, min_attrs=5):
            # Clean up saved file
            save_path.unlink(missing_ok=True)
            return None
        
        row = {
            "filename": save_filename,
            "safebooru_id": post_id,
            "original_width": w,
            "original_height": h,
            "raw_tags": tag_string
        }
        row.update(attributes)
        return row
    
    except Exception:
        return None


# ── Main Scraper ───────────────────────────────────────────────────────────

def scrape(num_images: int, output_dir: str, delay: float = 1.0, workers: int = 8):
    """
    Main scraping loop.
    - Fetches metadata pages sequentially (1000 posts per call)
    - Downloads images in parallel with a thread pool
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata_path = output_path.parent / "metadata.csv"
    
    search_tags = build_search_tags()
    print(f"Search tags: {search_tags}")
    print(f"Target: {num_images} images")
    print(f"Output: {output_path}")
    print(f"Workers: {workers} threads")
    print(f"Posts per API call: 1000")
    print(f"Tracking {len(ALL_ATTRIBUTE_TAGS)} attributes across {len(ATTRIBUTE_TAGS)} categories\n")
    
    # Prepare CSV
    fieldnames = ["filename", "safebooru_id", "original_width", "original_height"] + ALL_ATTRIBUTE_TAGS
    
    # Check for existing progress
    existing_ids = set()
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            existing_header = f.readline().strip().split(",")
            if existing_header != fieldnames:
                raise RuntimeError(
                    f"Schema mismatch. Header has {len(existing_header)} cols, "
                    f"code expects {len(fieldnames)}. Delete metadata.csv to rebuild."
                )
            
        with open(metadata_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row["safebooru_id"])
        print(f"Found {len(existing_ids)} existing images, resuming...\n")
        # csv_file = open(metadata_path, "a", newline="")
        # writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    else:
        with open(metadata_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # Thread-safe CSV writing
    csv_lock = threading.Lock()
    
    downloaded = len(existing_ids)
    page = 0
    consecutive_empty = 0
    pbar = tqdm(total=num_images, initial=downloaded, desc="Downloading")
    
    try:
        while downloaded < num_images:
            # ── Step 1: Fetch metadata page (1000 posts) ───────────────
            print(f"\n  Fetching page {page} (up to 1000 posts)...")
            posts = fetch_page(page, search_tags, limit=1000)
            
            if not posts:
                consecutive_empty += 1
                if consecutive_empty > 3:
                    print("\nNo more results from Safebooru. Stopping.")
                    break
                page += 1
                time.sleep(delay)
                continue
            
            consecutive_empty = 0
            print(f"  Got {len(posts)} posts. Filtering and downloading...")
            
            # ── Step 2: Filter out already-downloaded posts ────────────
            new_posts = []
            for post in posts:
                post_id = str(post.get("id", ""))
                if post_id not in existing_ids:
                    # Pre-check attributes before downloading
                    tag_string = post.get("tags", "")
                    attributes = extract_attributes(tag_string)
                    if has_minimum_attributes(attributes, min_attrs=5):
                        new_posts.append(post)
                        existing_ids.add(post_id)  # Reserve ID
            
            if not new_posts:
                page += 1
                continue
            
            remaining = num_images - downloaded
            new_posts = new_posts[:remaining]
            
            # ── Step 3: Download images in parallel ────────────────────
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(download_single_image, post, output_path): post
                    for post in new_posts
                }
                
                for future in as_completed(futures):
                    row = future.result()
                    if row is not None:
                        with csv_lock:
                            append_row(metadata_path, fieldnames, row)
                        downloaded += 1
                        pbar.update(1)
                        
                        if downloaded >= num_images:
                            break
            
            page += 1
            # Small delay between page fetches to be respectful
            time.sleep(delay)
    
    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Saved {downloaded} images so far.")
    finally:
        csv_file.close()
        pbar.close()
    
    print(f"\nDone! Downloaded {downloaded} images.")
    print(f"Metadata saved to {metadata_path}")
    print(f"Images saved to {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safebooru Kid-Friendly Cartoon Scraper")
    parser.add_argument("--num_images", type=int, default=10000,
                        help="Number of images to download (default: 10000)")
    parser.add_argument("--output_dir", type=str, default="./raw_images",
                        help="Directory to save images (default: ./raw_images)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API page fetches in seconds (default: 1.0)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of download threads (default: 8)")
    args = parser.parse_args()
    
    scrape(args.num_images, args.output_dir, args.delay, args.workers)
