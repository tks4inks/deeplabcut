"""video_pipeline: preprocess raw videos and run a single-animal DeepLabCut
pose-estimation pipeline to extract mouse XY movement.

Two independent stages live here:

* ``video_pipeline.probe`` / ``video_pipeline.standardize`` -- batch
  grayscale conversion, resolution standardization, and frame-rate
  downsampling for a folder of raw videos (no DeepLabCut dependency,
  no GPU required).
* ``video_pipeline.dlc_pipeline`` -- thin orchestration around the
  ``deeplabcut`` package (project creation -> labeling -> training ->
  inference -> XY export) for the standardized videos produced above.

See WORKFLOW.md at the repo root for the recommended end-to-end order
of operations and efficiency notes.
"""
