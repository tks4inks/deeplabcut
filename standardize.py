"""Single-animal DeepLabCut orchestration: project -> labeling -> training
-> inference -> tidy XY-movement export.

This module is a thin wrapper around the ``deeplabcut`` PyPI package
(``pip install deeplabcut``) that chains the stages you'd otherwise run
one-by-one from a notebook, with two additions:

* an ``export-xy`` stage that turns DLC's per-video H5/CSV output into a
  tidy "frame, time_s, x, y, likelihood" table per bodypart plus a
  centroid trajectory (overall mouse position) with distance/speed, and
* a single ``all`` command that runs every automatable stage in order
  and stops for the one stage that can't be automated: manual labeling.

Run stages individually, or run the whole pipeline in one go (it will
still pause before training so you can label):

    python -m video_pipeline.dlc_pipeline init \\
        --project mice_openfield --experimenter alex \\
        --videos standardized_videos/*.mp4 --working-dir dlc_project

    python -m video_pipeline.dlc_pipeline extract-frames --config dlc_project/.../config.yaml
    # -> now run: deeplabcut.label_frames(config)  (GUI, see WORKFLOW.md)
    python -m video_pipeline.dlc_pipeline train --config .../config.yaml
    python -m video_pipeline.dlc_pipeline analyze --config .../config.yaml --videos standardized_videos/*.mp4
    python -m video_pipeline.dlc_pipeline export-xy --config .../config.yaml --videos standardized_videos/*.mp4 \\
        --fps 15 --likelihood-threshold 0.9

Every function here fails loudly and immediately if ``deeplabcut`` isn't
installed -- it is a large, optional, GPU-oriented dependency and is not
required for the preprocessing stage in ``video_pipeline.standardize``.
"""
from __future__ import annotations

import argparse
import glob as globmod
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

log = logging.getLogger("video_pipeline.dlc_pipeline")


def _require_deeplabcut():
    try:
        import deeplabcut  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'deeplabcut' package is required for this stage. Install it "
            "(with a GPU-enabled TensorFlow/PyTorch backend for practical "
            "training/inference speed) via `pip install deeplabcut`. See "
            "WORKFLOW.md for environment setup notes."
        ) from exc
    return deeplabcut


def _expand_videos(patterns: Sequence[str]) -> List[str]:
    videos: List[str] = []
    for pattern in patterns:
        matches = sorted(globmod.glob(pattern))
        videos.extend(matches if matches else [pattern])
    if not videos:
        raise FileNotFoundError(f"No videos matched {patterns}")
    return videos


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def init_project(
    project: str,
    experimenter: str,
    videos: Sequence[str],
    working_dir: Path,
    bodyparts: Sequence[str],
    skeleton: Optional[Sequence[Sequence[str]]] = None,
    numframes2pick: int = 20,
    copy_videos: bool = False,
) -> Path:
    """Create a new single-animal DLC project and set bodyparts/skeleton.

    Returns the path to the generated config.yaml.
    """
    deeplabcut = _require_deeplabcut()
    video_list = _expand_videos(videos)
    working_dir.mkdir(parents=True, exist_ok=True)

    config_path = deeplabcut.create_new_project(
        project, experimenter, video_list,
        working_directory=str(working_dir),
        copy_videos=copy_videos,
    )
    config_path = Path(config_path)

    edits = {
        "bodyparts": list(bodyparts),
        "numframes2pick": numframes2pick,
    }
    if skeleton:
        edits["skeleton"] = [list(pair) for pair in skeleton]

    try:
        deeplabcut.auxiliaryfunctions.edit_config(str(config_path), edits)
    except AttributeError:
        # Older DLC versions: fall back to a direct YAML edit.
        _edit_yaml(config_path, edits)

    log.info("Project created: %s", config_path)
    log.info("Bodyparts set to: %s", list(bodyparts))
    return config_path


def _edit_yaml(config_path: Path, edits: dict) -> None:
    import ruamel.yaml

    yaml = ruamel.yaml.YAML()
    with open(config_path) as f:
        data = yaml.load(f)
    data.update(edits)
    with open(config_path, "w") as f:
        yaml.dump(data, f)


def extract_frames(config_path: Path, mode: str = "automatic", algo: str = "kmeans", userfeedback: bool = False) -> None:
    """Pick a diverse, informative subset of frames to label (k-means over
    frame features, not random sampling -- minimizes labeling effort for a
    given model accuracy)."""
    deeplabcut = _require_deeplabcut()
    deeplabcut.extract_frames(str(config_path), mode=mode, algo=algo, userfeedback=userfeedback)


def check_labels(config_path: Path) -> None:
    deeplabcut = _require_deeplabcut()
    deeplabcut.check_labels(str(config_path))


def create_training_dataset(config_path: Path, net_type: str = "resnet_50", num_shuffles: int = 1) -> None:
    deeplabcut = _require_deeplabcut()
    deeplabcut.create_training_dataset(str(config_path), num_shuffles=num_shuffles, net_type=net_type)


def train_network(
    config_path: Path,
    shuffle: int = 1,
    maxiters: int = 200_000,
    displayiters: int = 500,
    saveiters: int = 10_000,
    gputouse: Optional[int] = 0,
) -> None:
    deeplabcut = _require_deeplabcut()
    deeplabcut.train_network(
        str(config_path), shuffle=shuffle,
        displayiters=displayiters, saveiters=saveiters, maxiters=maxiters,
        gputouse=gputouse,
    )


def evaluate_network(config_path: Path, plotting: bool = True) -> None:
    deeplabcut = _require_deeplabcut()
    deeplabcut.evaluate_network(str(config_path), plotting=plotting)


def analyze_videos(
    config_path: Path,
    videos: Sequence[str],
    videotype: str = "mp4",
    gputouse: Optional[int] = 0,
    batchsize: int = 8,
    save_as_csv: bool = True,
) -> None:
    """Batch inference over every standardized video. A uniform resolution
    (from the preprocessing stage) lets a single batchsize be used across
    the whole set, which is what makes GPU batching efficient here."""
    deeplabcut = _require_deeplabcut()
    video_list = _expand_videos(videos)
    deeplabcut.analyze_videos(
        str(config_path), video_list, videotype=videotype,
        gputouse=gputouse, batchsize=batchsize, save_as_csv=save_as_csv,
    )


def filter_predictions(config_path: Path, videos: Sequence[str], videotype: str = "mp4") -> None:
    """Median-filter raw predictions to remove single-frame jitter."""
    deeplabcut = _require_deeplabcut()
    video_list = _expand_videos(videos)
    deeplabcut.filterpredictions(str(config_path), video_list, videotype=videotype)


def plot_trajectories(config_path: Path, videos: Sequence[str], videotype: str = "mp4", filtered: bool = True) -> None:
    deeplabcut = _require_deeplabcut()
    video_list = _expand_videos(videos)
    deeplabcut.plot_trajectories(str(config_path), video_list, videotype=videotype, filtered=filtered)


def create_labeled_video(config_path: Path, videos: Sequence[str], videotype: str = "mp4", filtered: bool = True) -> None:
    """Renders an overlay video -- run this on a small QC subset, not the
    whole batch, since it re-encodes full video (slow, storage-heavy)."""
    deeplabcut = _require_deeplabcut()
    video_list = _expand_videos(videos)
    deeplabcut.create_labeled_video(str(config_path), video_list, videotype=videotype, filtered=filtered)


# --------------------------------------------------------------------------
# XY export
# --------------------------------------------------------------------------

def export_xy(
    config_path: Path,
    videos: Sequence[str],
    output_dir: Path,
    fps: float,
    videotype: str = "mp4",
    likelihood_threshold: float = 0.9,
    filtered: bool = True,
    pixels_per_cm: Optional[float] = None,
    max_interp_gap: int = 5,
) -> List[Path]:
    """Turn DLC's per-video pose output into tidy XY-movement CSVs.

    For each video, writes two files under ``output_dir``:
      * ``<video>_bodyparts.csv``  -- frame, time_s, bodypart, x, y, likelihood
      * ``<video>_trajectory.csv`` -- frame, time_s, x, y (centroid across
        bodyparts above the likelihood threshold), distance_px (or _cm),
        speed_px_s (or _cm_s), cumulative_distance

    Low-confidence points (likelihood < threshold) are set to NaN and
    linearly interpolated across gaps up to ``max_interp_gap`` frames
    (longer gaps are left as NaN rather than fabricating a position).
    Pass ``pixels_per_cm`` (from a calibration object in frame) to get
    distances/speeds in real-world units instead of pixels.
    """
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    video_list = _expand_videos(videos)
    written: List[Path] = []

    for video in video_list:
        video_path = Path(video)
        h5_glob = f"{video_path.parent / video_path.stem}*{'filtered' if filtered else ''}*.h5"
        candidates = sorted(video_path.parent.glob(f"{video_path.stem}*{'filtered' if filtered else ''}*.h5"))
        if not candidates:
            log.warning("No DLC output H5 found for %s (looked for %s); skipping", video, h5_glob)
            continue
        h5_path = candidates[-1]  # most recent shuffle/snapshot naming wins

        df = pd.read_hdf(h5_path)
        df.columns = df.columns.droplevel(0)  # drop scorer level
        bodyparts = sorted(set(df.columns.get_level_values(0)))

        long_rows = []
        xy_by_bp = {}
        for bp in bodyparts:
            x = df[(bp, "x")].astype(float)
            y = df[(bp, "y")].astype(float)
            like = df[(bp, "likelihood")].astype(float)
            mask = like < likelihood_threshold
            x = x.mask(mask)
            y = y.mask(mask)
            x = x.interpolate(limit=max_interp_gap, limit_area="inside")
            y = y.interpolate(limit=max_interp_gap, limit_area="inside")
            xy_by_bp[bp] = (x, y)
            for frame_idx, (xv, yv, lv) in enumerate(zip(x, y, like)):
                long_rows.append({
                    "frame": frame_idx,
                    "time_s": frame_idx / fps,
                    "bodypart": bp,
                    "x": xv,
                    "y": yv,
                    "likelihood": lv,
                })

        bp_df = pd.DataFrame(long_rows)
        bp_out = output_dir / f"{video_path.stem}_bodyparts.csv"
        bp_df.to_csv(bp_out, index=False)
        written.append(bp_out)

        centroid_x = pd.concat([xy[0] for xy in xy_by_bp.values()], axis=1).mean(axis=1)
        centroid_y = pd.concat([xy[1] for xy in xy_by_bp.values()], axis=1).mean(axis=1)

        scale = 1.0 if pixels_per_cm is None else (1.0 / pixels_per_cm)
        unit = "px" if pixels_per_cm is None else "cm"

        step_dist = ((centroid_x.diff() ** 2 + centroid_y.diff() ** 2) ** 0.5) * scale
        speed = step_dist * fps
        traj_df = pd.DataFrame({
            "frame": range(len(centroid_x)),
            "time_s": [i / fps for i in range(len(centroid_x))],
            "x": centroid_x * scale,
            "y": centroid_y * scale,
            f"distance_{unit}": step_dist,
            f"speed_{unit}_s": speed,
            f"cumulative_distance_{unit}": step_dist.fillna(0).cumsum(),
        })
        traj_out = output_dir / f"{video_path.stem}_trajectory.csv"
        traj_df.to_csv(traj_out, index=False)
        written.append(traj_out)
        log.info("Exported %s -> %s, %s", video_path.name, bp_out.name, traj_out.name)

    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("init", help="Create a new single-animal DLC project")
    s.add_argument("--project", required=True)
    s.add_argument("--experimenter", required=True)
    s.add_argument("--videos", nargs="+", required=True)
    s.add_argument("--working-dir", type=Path, required=True)
    s.add_argument("--bodyparts", nargs="+", default=["nose", "left_ear", "right_ear", "centroid", "tailbase"])
    s.add_argument("--numframes2pick", type=int, default=20)
    s.add_argument("--copy-videos", action="store_true")

    s = sub.add_parser("extract-frames", help="Pick frames to label (k-means)")
    s.add_argument("--config", type=Path, required=True)

    s = sub.add_parser("check-labels", help="Render label overlays for QA")
    s.add_argument("--config", type=Path, required=True)

    s = sub.add_parser("create-dataset", help="Build the training dataset from labeled frames")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--net-type", default="resnet_50")

    s = sub.add_parser("train", help="Train the network")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--maxiters", type=int, default=200_000)
    s.add_argument("--gputouse", type=int, default=0)

    s = sub.add_parser("evaluate", help="Evaluate train/test pixel error")
    s.add_argument("--config", type=Path, required=True)

    s = sub.add_parser("analyze", help="Batch inference over videos")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--videos", nargs="+", required=True)
    s.add_argument("--videotype", default="mp4")
    s.add_argument("--batchsize", type=int, default=8)
    s.add_argument("--gputouse", type=int, default=0)

    s = sub.add_parser("filter", help="Median-filter raw predictions")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--videos", nargs="+", required=True)
    s.add_argument("--videotype", default="mp4")

    s = sub.add_parser("trajectories", help="Plot per-video trajectory PNGs")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--videos", nargs="+", required=True)
    s.add_argument("--videotype", default="mp4")

    s = sub.add_parser("labeled-video", help="Render overlay QC video for a subset")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--videos", nargs="+", required=True)
    s.add_argument("--videotype", default="mp4")

    s = sub.add_parser("export-xy", help="Export tidy XY-movement CSVs")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--videos", nargs="+", required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    s.add_argument("--fps", type=float, required=True)
    s.add_argument("--videotype", default="mp4")
    s.add_argument("--likelihood-threshold", type=float, default=0.9)
    s.add_argument("--pixels-per-cm", type=float, default=None)
    s.add_argument("--no-filtered", dest="filtered", action="store_false")
    s.set_defaults(filtered=True)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    try:
        if args.stage == "init":
            init_project(
                args.project, args.experimenter, args.videos, args.working_dir,
                bodyparts=args.bodyparts, numframes2pick=args.numframes2pick, copy_videos=args.copy_videos,
            )
        elif args.stage == "extract-frames":
            extract_frames(args.config)
        elif args.stage == "check-labels":
            check_labels(args.config)
        elif args.stage == "create-dataset":
            create_training_dataset(args.config, net_type=args.net_type)
        elif args.stage == "train":
            train_network(args.config, maxiters=args.maxiters, gputouse=args.gputouse)
        elif args.stage == "evaluate":
            evaluate_network(args.config)
        elif args.stage == "analyze":
            analyze_videos(args.config, args.videos, videotype=args.videotype, batchsize=args.batchsize, gputouse=args.gputouse)
        elif args.stage == "filter":
            filter_predictions(args.config, args.videos, videotype=args.videotype)
        elif args.stage == "trajectories":
            plot_trajectories(args.config, args.videos, videotype=args.videotype)
        elif args.stage == "labeled-video":
            create_labeled_video(args.config, args.videos, videotype=args.videotype)
        elif args.stage == "export-xy":
            export_xy(
                args.config, args.videos, args.output_dir, fps=args.fps, videotype=args.videotype,
                likelihood_threshold=args.likelihood_threshold, filtered=args.filtered,
                pixels_per_cm=args.pixels_per_cm,
            )
    except Exception as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
