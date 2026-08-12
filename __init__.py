"""Unit tests for the pure-logic parts of video_pipeline.standardize.

These deliberately avoid touching cv2/ffmpeg/actual video files so they
run anywhere (including environments without ffmpeg or opencv installed).
Run with: python -m unittest tests.test_standardize -v
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from video_pipeline.probe import VideoInfo
from video_pipeline.standardize import (
    DEFAULT_EXTENSIONS,
    TargetSpec,
    build_ffmpeg_command,
    build_output_path,
    compute_target_spec,
    discover_videos,
)


def make_info(name, width, height, fps):
    return VideoInfo(path=Path(name), width=width, height=height, fps=fps, frame_count=None, duration=None)


class TargetSpecTests(unittest.TestCase):
    def test_rejects_non_positive_dims(self):
        with self.assertRaises(ValueError):
            TargetSpec(width=0, height=480, fps=30)
        with self.assertRaises(ValueError):
            TargetSpec(width=640, height=480, fps=0)

    def test_rounds_odd_dimensions_down_to_even(self):
        spec = TargetSpec(width=641, height=481, fps=30)
        self.assertEqual((spec.width, spec.height), (640, 480))


class ComputeTargetSpecTests(unittest.TestCase):
    def test_uses_minimum_across_batch_when_unspecified(self):
        infos = [
            make_info("a.mp4", 1920, 1080, 30.0),
            make_info("b.mp4", 640, 480, 60.0),
            make_info("c.mp4", 1280, 720, 15.0),
        ]
        spec = compute_target_spec(infos)
        self.assertEqual((spec.width, spec.height), (640, 480))
        self.assertEqual(spec.fps, 15.0)

    def test_explicit_values_override_batch(self):
        infos = [make_info("a.mp4", 1920, 1080, 30.0)]
        spec = compute_target_spec(infos, width=100, height=100, fps=5)
        self.assertEqual((spec.width, spec.height, spec.fps), (100, 100, 5.0))

    def test_ignores_zero_fps_when_deriving_default(self):
        infos = [make_info("a.mp4", 640, 480, 0.0), make_info("b.mp4", 640, 480, 24.0)]
        spec = compute_target_spec(infos)
        self.assertEqual(spec.fps, 24.0)

    def test_raises_on_empty_input(self):
        with self.assertRaises(ValueError):
            compute_target_spec([])

    def test_raises_when_no_video_has_usable_fps(self):
        infos = [make_info("a.mp4", 640, 480, 0.0)]
        with self.assertRaises(ValueError):
            compute_target_spec(infos)


class DiscoverVideosTests(unittest.TestCase):
    def test_finds_only_known_extensions_non_recursive(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp4").touch()
            (root / "b.txt").touch()
            (root / "c.AVI").touch()  # case-insensitive match
            sub = root / "nested"
            sub.mkdir()
            (sub / "d.mp4").touch()

            found = discover_videos(root, DEFAULT_EXTENSIONS, recursive=False)
            self.assertEqual([p.name for p in found], ["a.mp4", "c.AVI"])

    def test_recursive_includes_nested_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "nested"
            sub.mkdir()
            (sub / "d.mp4").touch()

            found = discover_videos(root, DEFAULT_EXTENSIONS, recursive=True)
            self.assertEqual([p.name for p in found], ["d.mp4"])

    def test_raises_on_missing_directory(self):
        with self.assertRaises(NotADirectoryError):
            discover_videos(Path("/no/such/dir/should/exist"))


class BuildOutputPathTests(unittest.TestCase):
    def test_preserves_relative_layout_and_swaps_extension(self):
        input_dir = Path("/data/raw")
        output_dir = Path("/data/std")
        src = Path("/data/raw/session1/mouse.avi")
        dst = build_output_path(src, input_dir, output_dir, ".mp4")
        self.assertEqual(dst, Path("/data/std/session1/mouse.mp4"))


class BuildFfmpegCommandTests(unittest.TestCase):
    def test_includes_grayscale_filter_when_enabled(self):
        spec = TargetSpec(width=640, height=480, fps=15)
        cmd = build_ffmpeg_command(Path("in.mp4"), Path("out.mp4"), spec, grayscale=True)
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("format=gray", vf)
        self.assertIn("fps=15", vf)
        self.assertIn("scale=640:480", vf)

    def test_omits_grayscale_filter_when_disabled(self):
        spec = TargetSpec(width=640, height=480, fps=15)
        cmd = build_ffmpeg_command(Path("in.mp4"), Path("out.mp4"), spec, grayscale=False)
        vf = cmd[cmd.index("-vf") + 1]
        self.assertNotIn("format=gray", vf)

    def test_drops_audio_stream(self):
        spec = TargetSpec(width=640, height=480, fps=15)
        cmd = build_ffmpeg_command(Path("in.mp4"), Path("out.mp4"), spec)
        self.assertIn("-an", cmd)


if __name__ == "__main__":
    unittest.main()
