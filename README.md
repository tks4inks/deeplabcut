# deeplabcut

Tools for turning raw mouse-tracking videos into XY movement data:

1. **`video_pipeline/standardize.py`** — batch-convert raw videos to
   grayscale, a single standardized resolution, and a lower/uniform
   frame rate (the main lever for reducing drive storage).
2. **`video_pipeline/dlc_pipeline.py`** — orchestrates a single-animal
   [DeepLabCut](https://deeplabcut.github.io) project (create → label →
   train → analyze) and exports tidy per-frame XY / trajectory CSVs.

Start with **[WORKFLOW.md](WORKFLOW.md)** for the full step-by-step,
efficiency-minded pipeline and copy-pasteable commands.

## Quick start

```bash
pip install -r video_pipeline/requirements.txt   # + ffmpeg binary, see WORKFLOW.md

# 1. grayscale + standardize + downsample fps
python -m video_pipeline.standardize \
    --input-dir raw_videos/ --output-dir standardized_videos/ --fps 15

# 2. run the DeepLabCut pipeline (see WORKFLOW.md for every stage)
python -m video_pipeline.dlc_pipeline --help
```

## Tests

```bash
python -m unittest tests.test_standardize -v
```
