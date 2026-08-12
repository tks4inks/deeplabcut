# Mouse XY-tracking workflow: raw video → grayscale/standardized/downsampled → DeepLabCut → XY movement CSVs

This repo currently ships two pieces of code, both under `video_pipeline/`:

* `video_pipeline/standardize.py` — batch grayscale + resolution + fps
  standardization of raw videos. No GPU, no DeepLabCut required.
* `video_pipeline/dlc_pipeline.py` — orchestration around the
  `deeplabcut` package: project creation → labeling → training →
  inference → tidy XY-movement export. Requires `pip install
  deeplabcut` and, in practice, a GPU.

Everything below assumes **one mouse per video** (single-animal DLC —
simpler, faster to train, and no cross-frame identity/tracklet-stitching
step is needed).

Run all of this **on a machine with your video files and, for training/
inference, a GPU** — this sandbox has neither ffmpeg, OpenCV, nor
`deeplabcut` installed, and no video files exist here to run against.
The code has been unit-tested for its pure logic (`tests/`) but the
video/DLC stages themselves need to be run in your real environment.

## Why this order, and why it's efficient

1. **Preprocess before anything else.** Grayscale + a lower, uniform fps
   + a capped resolution typically shrinks a raw video folder 5–20x —
   do this first if drive storage is the constraint, so you're not
   waiting on (or paying for) slow I/O against huge raw files in every
   later step. It also means DLC only ever sees one resolution/fps
   across the whole project, which is what makes k-means frame
   selection, labeling, and batched GPU inference all straightforward.
2. **Downsampling fps is the single biggest storage lever** for typical
   open-field/home-cage mouse tracking: file size scales ~linearly with
   fps, and locomotion behavior rarely needs more than 15–30 fps to
   track accurately (compare that to your source fps — if you're
   recording at 60/120 fps for some other reason, e.g. to catch fast
   head-turns, keep it higher; otherwise drop it). Fewer fps also means
   fewer frames to label and fewer frames to run inference over later.
3. **Grayscale** roughly halves per-frame memory/bandwidth (1 channel
   vs. 3) and mouse tracking essentially never depends on color, so
   there's no accuracy trade-off, only speed/storage wins.
4. **Delete or archive raw videos only after verifying standardized
   output** (spot-check a few with a player) — the pipeline never
   deletes your originals itself.

## Step-by-step

### 0. Environment setup (once, on your GPU machine)

```bash
pip install -r video_pipeline/requirements.txt
# ffmpeg binary (preferred backend for stage 1) — pick one:
sudo apt-get install ffmpeg        # Debian/Ubuntu
# or: brew install ffmpeg          # macOS
# or: conda install -c conda-forge ffmpeg
```

`deeplabcut` pulls in a large deep-learning backend; for anything past a
handful of frames you want a CUDA-enabled GPU build — see
https://deeplabcut.github.io for the current recommended install for
your OS/GPU.

### 1. Grayscale + standardize + downsample fps

```bash
python -m video_pipeline.standardize \
    --input-dir raw_videos/ \
    --output-dir standardized_videos/ \
    --fps 15 \
    --manifest standardized_videos/manifest.csv \
    --max-workers 4 \
    -v
```

- Omit `--width/--height/--fps` and the tool auto-picks the *minimum*
  resolution/fps observed across your input batch (never upscales or
  interpolates frames — only ever downsamples). Pass them explicitly if
  you know what you want, e.g. `--fps 15` to force 15 fps regardless of
  source.
- Run with `--dry-run` first to see the chosen target spec and per-file
  probe results before committing to the conversion.
- `--max-workers` controls how many ffmpeg jobs run in parallel — scale
  to your CPU core count.
- Output keeps the same relative folder layout as the input, so
  subfolders (e.g. per session/cohort) are preserved.
- Check `standardized_videos/manifest.csv` for a per-file source→output
  record (useful for provenance and for catching any file that failed
  to convert).

**Storage tip:** once you've confirmed a handful of standardized outputs
look correct (play them back), you can move the raw originals to cold/
archival storage — the standardized set is what feeds everything below.

### 2. Create the DeepLabCut project

```bash
python -m video_pipeline.dlc_pipeline init \
    --project mice_openfield --experimenter <your_name> \
    --videos standardized_videos/*.mp4 \
    --working-dir dlc_project \
    --bodyparts nose left_ear right_ear centroid tailbase \
    --numframes2pick 20
```

Pick bodyparts that match what you actually need for "mouse XY
movement": `centroid` alone is enough for a locomotion/position trace;
add `nose`/`ears`/`tailbase` if you also want heading direction or
body-length-normalized speed. This writes `dlc_project/.../config.yaml`
— every later command takes `--config <that path>`.

### 3. Extract frames to label

```bash
python -m video_pipeline.dlc_pipeline extract-frames --config dlc_project/.../config.yaml
```

Uses k-means over frame content to pick diverse, informative frames
(not random sampling) — this minimizes how many frames you need to
hand-label for a given accuracy target. `--numframes2pick 20` (set in
step 2) times your number of videos is roughly how many frames you'll
label; 15–20 per video is a reasonable starting point for single-animal
tracking with a handful of bodyparts.

### 4. Label frames (manual — the one step that can't be automated)

```python
import deeplabcut
deeplabcut.label_frames("dlc_project/.../config.yaml")
```

This opens a GUI. Label every extracted frame consistently (same
bodypart definitions every time, mark occluded points as "not visible"
rather than guessing). Then QA:

```bash
python -m video_pipeline.dlc_pipeline check-labels --config dlc_project/.../config.yaml
```

which renders labeled overlays into `labeled-data/.../CollectedData_*_labeled/`
so you can eyeball mistakes before training on them.

### 5. Create the training dataset and train

```bash
python -m video_pipeline.dlc_pipeline create-dataset --config dlc_project/.../config.yaml --net-type resnet_50
python -m video_pipeline.dlc_pipeline train --config dlc_project/.../config.yaml --maxiters 200000
```

- `resnet_50` is a good default for single-animal tracking: fast to
  train, accurate enough for most locomotion/position tasks. Only
  reach for a heavier backbone if evaluation error is unacceptably
  high after checking your labels.
- Efficiency: this step is GPU-bound; a single mid-range GPU handles a
  single-animal resnet_50 project in a few hours for typical dataset
  sizes. Watch training loss via the `displayiters` log — you can often
  stop well before `maxiters` once loss plateaus (train from a saved
  snapshot with a smaller `--maxiters` if you want to extend later).

### 6. Evaluate

```bash
python -m video_pipeline.dlc_pipeline evaluate --config dlc_project/.../config.yaml
```

Reports train/test pixel error. If it's too high relative to your
video's scale (e.g. >1–2% of frame diagonal), go back to step 3 with
`mode="automatic", algo="uniform"` or extract *outlier* frames from a
first-pass `analyze_videos` run and relabel — cheaper than blindly
labeling more frames.

### 7. Batch-analyze every standardized video

```bash
python -m video_pipeline.dlc_pipeline analyze \
    --config dlc_project/.../config.yaml \
    --videos standardized_videos/*.mp4 \
    --batchsize 8
```

Because every video shares one resolution (step 1), a single
`--batchsize` works across the whole set — raise it until you're close
to GPU memory limits for faster throughput. This produces one H5/CSV of
raw per-frame bodypart coordinates + confidence per video.

### 8. Filter predictions

```bash
python -m video_pipeline.dlc_pipeline filter \
    --config dlc_project/.../config.yaml --videos standardized_videos/*.mp4
```

Median-filters out single-frame jitter/outliers before you compute
movement metrics from the trace.

### 9. Export tidy XY movement

```bash
python -m video_pipeline.dlc_pipeline export-xy \
    --config dlc_project/.../config.yaml \
    --videos standardized_videos/*.mp4 \
    --output-dir xy_results/ \
    --fps 15 \
    --likelihood-threshold 0.9
```

(Use the same `--fps` you standardized to in step 1 — that's each
video's true frame rate now, needed to convert frame index → seconds
and to turn per-frame displacement into speed.)

For each video this writes:
- `<video>_bodyparts.csv` — `frame, time_s, bodypart, x, y, likelihood`,
  one row per bodypart per frame (low-confidence points below the
  threshold are NaN'd and short gaps linearly interpolated; long gaps
  are left as NaN rather than invented).
- `<video>_trajectory.csv` — `frame, time_s, x, y, distance_px,
  speed_px_s, cumulative_distance_px` for the mouse's **centroid**
  (mean of all above-threshold bodyparts each frame) — this is the
  "mouse XY movement over time" trace most analyses actually want.

Pass `--pixels-per-cm <value>` (from a calibration object of known size
in your arena) to get distances/speeds in cm instead of pixels.

### 10. Spot-check visually (subset only)

```bash
python -m video_pipeline.dlc_pipeline trajectories --config dlc_project/.../config.yaml --videos standardized_videos/mouse01.mp4
python -m video_pipeline.dlc_pipeline labeled-video  --config dlc_project/.../config.yaml --videos standardized_videos/mouse01.mp4
```

Run `labeled-video` (it re-encodes a full overlay video — slow and
storage-heavy) on a handful of representative videos, not the whole
batch, purely as a visual sanity check that tracking looks right before
you trust the CSVs for the rest.

## End-to-end summary

```
raw_videos/  →  [standardize.py: grayscale + resize + fps↓]  →  standardized_videos/
                                                                        │
                                                    [dlc_pipeline.py: init → extract-frames]
                                                                        │
                                                      (manual) label_frames GUI
                                                                        │
                                              create-dataset → train → evaluate
                                                                        │
                                                      analyze → filter → export-xy
                                                                        │
                                                                 xy_results/*.csv
```

## Tests

`tests/test_standardize.py` covers the pure logic (target-spec
computation, file discovery, ffmpeg command construction, output path
layout) with no video/GPU dependencies — run it anywhere with:

```bash
python -m unittest tests.test_standardize -v
```
