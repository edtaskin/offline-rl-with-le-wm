import math
from pathlib import Path

import numpy as np


def save_evaluation_video(frames, video_path, fps=30):
    if video_path is None:
        return None
    if frames is None:
        frames = []
    elif isinstance(frames, np.ndarray):
        frames = list(frames)
    else:
        frames = list(frames)
    if not frames:
        print(f"No evaluation frames captured; skipping video save to {video_path}.")
        return None
    if fps <= 0:
        raise ValueError("video fps must be positive")

    output_path = Path(video_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def prepare_frame(frame, expected_shape=None):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("evaluation video frames must be RGB arrays with shape (height, width, 3)")
        if expected_shape is not None and frame.shape[:2] != expected_shape:
            raise ValueError("all evaluation video frames must have the same size")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    first_frame = prepare_frame(frames[0])
    height, width = first_frame.shape[:2]
    import cv2

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    try:
        writer.write(first_frame[..., ::-1])
        for frame in frames[1:]:
            writer.write(prepare_frame(frame, (height, width))[..., ::-1])
    finally:
        writer.release()
    print(f"Saved evaluation video to: {output_path}")
    return output_path


def is_video_file_path(path):
    return Path(path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


def combine_world_panel_videos(video_dir, output_path, fps=None):
    import cv2

    video_dir = Path(video_dir)
    output_path = Path(output_path)
    video_paths = sorted(
        video_dir.glob("env_*.mp4"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not video_paths:
        print(f"No swm.World env_*.mp4 videos found in {video_dir}; skipping combine.")
        return None

    captures = [cv2.VideoCapture(str(path)) for path in video_paths]
    writer = None
    try:
        first_frames = []
        for capture, path in zip(captures, video_paths):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read first frame from {path}")
            first_frames.append(frame)
        tile_h, tile_w = first_frames[0].shape[:2]
        source_fps = captures[0].get(cv2.CAP_PROP_FPS) or 15.0
        output_fps = float(fps or source_fps)
        cols = math.ceil(math.sqrt(len(video_paths)))
        rows = math.ceil(len(video_paths) / cols)
        grid_w = cols * tile_w
        grid_h = rows * tile_h
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (grid_w, grid_h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")
        last_frames = first_frames
        while True:
            canvas = np.full((grid_h, grid_w, 3), 250, dtype=np.uint8)
            for index, frame in enumerate(last_frames):
                row, col = divmod(index, cols)
                y0, x0 = row * tile_h, col * tile_w
                canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = frame
            writer.write(canvas)
            any_active = False
            next_frames = []
            for capture, last_frame in zip(captures, last_frames):
                ok, frame = capture.read()
                if ok:
                    any_active = True
                    next_frames.append(frame)
                else:
                    next_frames.append(last_frame)
            if not any_active:
                break
            last_frames = next_frames
    finally:
        for capture in captures:
            capture.release()
        if writer is not None:
            writer.release()
    print(f"Saved combined swm.World video to: {output_path}")
    return output_path
