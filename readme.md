# Data Pipeline for Conditional DDPM — Engineering Notes

**Project context:** Building a conditional denoising diffusion model from scratch to generate kid-friendly cartoon characters from multi-hot attribute vectors. This document covers the data acquisition and preprocessing phase — the unglamorous but consequential 80% of the work.

---

## What I built

A scraper and post-processing pipeline that pulls single-character illustrations from Safebooru, extracts a 135-dimensional attribute vector per image across 11 semantic categories (hair color, expression, accessories, pose, etc.), and produces a clean training set ready for diffusion model training.

**Final dataset:** ~19k images after MD5 + perceptual-hash deduplication, center-cropped and resized to 64×64, with attribute distributions analyzed and underrepresented tags pruned.

---

## Key technical decisions and the reasoning behind them

### Resolution: committed to 64×64

Pixel-space diffusion compute scales quadratically with side length. With <50k training images, 128×128 produces undertrained, blurry samples no matter how long you run. 256×256+ requires either millions of images or a latent-diffusion architecture (autoencoder + diffusion in latent space), which doubles project scope. 64×64 is the honest choice for the data and compute available.

### Cropping: center-crop over stretch or pad

Diffusion U-Nets require fixed-size square inputs. Three options exist:
- **Stretch:** distorts geometry; the model learns to generate distorted characters.
- **Pad-to-square:** wastes pixels on bars and forces the model to learn the bar pattern.
- **Center-crop:** preserves geometry, uses every pixel for content. Combined with an aspect-ratio filter (max 2.0), losses to off-center subjects are bounded.

Used Lanczos resampling (not bilinear) for the final downsample — sharper edges with no extra cost.

### Conditioning mechanism: AdaGN concat over cross-attention

A 135-dim multi-hot vector has no sequential or spatial structure. Cross-attention's strength is conditioning on structured inputs (token sequences, spatial maps). For fixed-length attribute vectors, projecting through an MLP and concatenating with the timestep embedding (fed via adaptive group normalization) is sufficient and avoids unnecessary parameters.

---

## Problems hit and how I resolved them

### 1. CSV schema drift between scraper runs

**Symptom:** After expanding the attribute tag list mid-project, `metadata.csv` had the old narrow header (~70 columns) but new rows trying to write 139 fields. Pandas loaded the file with "Unnamed: N" columns for everything past the original header.

**Root cause:** The resume logic opened the CSV in append mode without validating the existing header against the current schema.

**Fix:** Added a schema-version guard that compares the existing CSV header against the current `fieldnames` at startup and raises if they differ. Cheap insurance against silent data corruption when the schema evolves.

```python
if existing_header != fieldnames:
    raise RuntimeError(f"Schema mismatch: {len(existing_header)} vs {len(fieldnames)}")
```

### 2. Long-running file handle risked data loss on dirty exit

The original code held a single CSV file handle open for the entire (multi-hour) run with `flush()` after each write. `flush()` only pushes to the OS buffer — a hard kill (OOM, network loss, machine reboot) could still drop buffered rows or leave a half-written final line.

**Fix:** Switched to open-write-close per row inside a `csv_lock`. Adds ~35 seconds total over a 70k-row run versus durability against process death. Trivial cost, real benefit.

### 3. Scraper stopped at 19,461 / 70,000 target

**Diagnosis:** Hit Safebooru's hard pagination cap at page 200 (≈200k posts). The cap is a backend limitation of the Gelbooru-family search index, not a rate limit you can wait out.

**Compounded by:** A `-highres` exclusion filter that was systematically removing the most polished, popular images on the site. `highres` is a metadata tag for files above ~1600×1200 — exactly the images that downsample best to 64×64. I had originally added the filter thinking about download bandwidth; it was the wrong tool for that goal and was costing significant yield.

**Fix:**
- Removed `-highres` (kept content-based exclusions like `-nude`, `-blood`, `-weapon`).
- Loosened `min_attrs` from 5 to 4 — fewer richly-tagged images excluded, attribute signal still strong enough.
- Added `-4koma` to better catch multi-panel content that wasn't getting filtered by `-comic` alone.

**Lesson:** Distinguish content filters (worth the yield cost) from quality/metadata filters (need scrutiny). A negative tag's name doesn't tell you what it actually filters.

### 4. Deduplication needed two passes, not one

MD5 hashing only catches byte-identical files. Same image re-saved at different JPEG quality, with a watermark, or as a different file format produces a different MD5 — but is functionally a duplicate for training purposes.

**Fix:** Two-stage dedup:
- **MD5 hash** (fast, exact): catches literal reposts. Removed ~2%.
- **Perceptual hash (pHash)** (slower, structural): grayscales → resizes to 32×32 → DCT → keeps low-frequency 8×8 block → binarizes against median → 64-bit fingerprint. Catches re-encodings, watermarks, minor edits. Removed another ~5%.

Did both as post-processing scripts on the downloaded files rather than inline during scraping — cleaner separation, easier to debug, can be rerun independently.

### 5. Attribute distribution was severely long-tailed

With 135 binary attributes and 19k images, several tags had <50 examples while popular ones had thousands. Training conditioning on the rare ones would produce noise; the eval classifier would have no signal to score them on.

**Fix:** Computed per-attribute frequency counts, then:
- Dropped attributes with <200 examples entirely from the conditioning vector.
- Merged fine-grained variants where it made sense (`cat_tail` + `dog_tail` + `fox_tail` → `tail`).
- Flagged 200–500 example attributes as "low-confidence" for separate reporting in the eval phase.

Also computed pairwise correlation between remaining attributes to identify entangled pairs (`wand` ~ `witch_hat`, `wings` ~ `flying`). These get flagged in the disentanglement evaluation as "expected to be entangled given training data co-occurrence" rather than treated as model failures.

### 6. Train/val split needed stratification

Random 95/5 split left rare attributes with <10 validation examples — making the eval classifier's reported accuracy on those tags noisy enough to be meaningless.

**Fix:** Iterative stratified splitting (multilabel-aware) to ensure every attribute has a reasonable val-set count. Alternative for simpler cases: spot-check the random split and rerun with a different seed if rare attributes get unlucky.

---

## What I'd do differently next time

1. **Schema validation on day one.** The CSV schema bug cost an evening of confusion. Five lines of validation code at the start would have caught it immediately. This is now my default for any pipeline that writes structured output across runs.

2. **Pull from multiple narrow queries instead of one broad one.** Safebooru's pagination cap is per-query. Running `solo cat_ears`, `solo dragon`, `solo magical_girl` separately and deduping the union sidesteps the page-200 wall entirely. Would have hit the 70k target without compromising filter quality.

3. **Resize on save with original kept separately.** Storing 70k full-resolution PNGs is 100+ GB. Storing them at 512×512 cap is ~5 GB and gives flexibility to experiment with 128×128 later without rescraping.

4. **Treat data acquisition as iterative.** First run with conservative filters and small target; analyze the distribution; adjust filters; run again. I conflated "scrape" and "scrape *correctly*" — they're different problems.

---

## Skills demonstrated

**Engineering:** Multithreaded scraping with thread-safe state, resumable long-running pipelines, schema versioning, robust file I/O patterns under concurrency.

**ML data pipeline:** Multi-stage deduplication (byte-level + perceptual), attribute distribution analysis, stratified multilabel splitting, co-occurrence analysis for downstream evaluation design.

**Judgment:** Distinguishing content vs metadata filters, recognizing when a target dataset size isn't worth the marginal effort, identifying which preprocessing decisions affect model behavior (resolution, cropping, conditioning) versus which are cosmetic.

**Debugging:** Diagnosing schema drift from "Unnamed: N" symptoms, identifying pagination caps from termination patterns, distinguishing transient API errors from end-of-results signals.


---

# Appendix: Deduplication and Concurrency

After scraping, two cleanup passes ran on the dataset: exact-duplicate detection via MD5 and near-duplicate detection via perceptual hash (pHash). Both involved choices about concurrency that turned out to be non-obvious.

## Concurrency: matching the model to the workload

**MD5 hashing is I/O-bound.** Reading the file dominates; the hash itself is microseconds. Threads work well here — the GIL releases during disk reads, so a `ThreadPoolExecutor` with 16 workers gives near-linear speedup with no setup cost. For 19k files, ~30 seconds.

**pHash is mixed I/O and CPU.** PIL decodes the PNG (releases GIL — parallelizes), then the resize + DCT + binarize pipeline runs in pure Python and holds the GIL. Threading still helps (~4–6x) because PNG decode is a meaningful chunk of work, but it doesn't fully utilize cores. For true parallelism, `ProcessPoolExecutor` with workers equal to `cpu_count() - 1` runs each pHash on its own core and finishes ~2x faster than threading.

**Lesson:** "Make it concurrent" isn't a single decision. Threads for I/O-bound, processes for CPU-bound, neither for tasks that finish in seconds anyway. For a 19k-file one-shot job, threading was the right call — process pools added complexity (worker functions must live in `.py` files for pickling, notebook cells don't work) for marginal gain.

## Notebook-specific gotchas

`multiprocessing.Pool` and `ProcessPoolExecutor` need worker functions to be importable. Functions defined in notebook cells aren't — they live in kernel memory, not on disk. Workers spawn, fail to import, and the notebook either errors with `AttributeError: Can't get attribute` or silently hangs. Fix: move worker functions to a `.py` file next to the notebook.

Threading has no such issue and works identically in scripts and notebooks. Another reason to default to threads when the workload allows.

## Two-stage dedup: inspect, then act

Built `view_duplicates(df, hash_col, img_dir)` and `remove_duplicates(df, hash_col, img_dir)` as separate functions. `view` finds groups, prints filenames, and renders a few groups inline with matplotlib for visual confirmation. `remove` does the actual deletion, with a `dry_run` flag for one more safety check.

Same functions handle both MD5 and pHash by parameterizing the column name — no copy-paste between dedup passes.

The pattern matters: any operation that deletes data from disk should be split into a "show me what would happen" call and a "do it" call. Not because the dedup logic is wrong, but because seeing the actual files being merged catches edge cases (e.g., are these really duplicates, or just visually similar characters?) that the metric alone misses.

## Pass order matters

Run MD5 first, then pHash. MD5 matches are a strict subset of pHash matches (byte-identical files have identical pHashes), so MD5 eliminates the easy cases in O(n). pHash handles the structural near-duplicates afterward on the smaller surviving set.

For pHash, sort by resolution descending before deduping with `keep="first"` — that way when two near-duplicates are found, the higher-resolution copy survives. Doesn't matter for MD5 (byte-identical means same resolution).

## Hamming-distance pHash dedup: considered, deferred

Exact-match pHash catches duplicates whose 64-bit fingerprints are identical. Hamming distance ≤ 5 catches additional near-duplicates that differ by a few bits (cropped versions, color edits). Skipped for this project: the marginal gain on a 19k-image set isn't worth the O(n²) comparison cost or the false-positive risk (two distinct characters in similar poses can land within 5 bits). Worth revisiting if memorization shows up in trained samples.

## Minor lessons

- `tqdm.notebook` requires `ipywidgets`. Plain `from tqdm import tqdm` works everywhere and avoids one dependency.
- `ex.map` with `chunksize=32–64` is the right default for short tasks across processes — without chunking, IPC overhead dominates.
- When using process pools, sanity-check parallelism is real: open a system monitor and confirm multiple Python processes are pinning cores. If only one is, the worker function isn't being parallelized correctly.

## Skills demonstrated (additions)

**Concurrency:** Distinguishing I/O-bound from CPU-bound workloads, choosing between threading and multiprocessing accordingly, understanding GIL behavior under different operations, diagnosing notebook-specific multiprocessing failures.

**Defensive design:** Two-stage destructive operations (inspect → confirm → act), parameterized utility functions to avoid duplication across similar passes, ordering operations by cost (cheap exact matches before expensive fuzzy ones).