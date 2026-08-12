# Preprocessing stage (video_pipeline.standardize) -- no GPU needed.
# ffmpeg is the preferred backend; install the binary separately
# (e.g. `apt install ffmpeg` / `brew install ffmpeg` / conda-forge).
# OpenCV is used automatically as a fallback if ffmpeg isn't on PATH,
# and by video_pipeline.probe as a fallback for ffprobe.
opencv-python-headless>=4.8

# DLC pipeline stage (video_pipeline.dlc_pipeline) -- GPU strongly
# recommended for train/analyze. `deeplabcut` pulls in its own deep
# learning backend; install a CUDA-enabled build for that backend
# separately per https://deeplabcut.github.io if you have a GPU.
deeplabcut>=2.3
pandas>=1.5
tables>=3.7          # pytables, required by pandas.read_hdf
ruamel.yaml>=0.17     # only used as a config.yaml edit fallback on old DLC
