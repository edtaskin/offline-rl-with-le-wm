from pathlib import Path

import numpy as np

def _upscale(frame, resolution):
    frame = np.asarray(frame)
    if resolution <= 0 or frame.shape[:2] == (resolution, resolution):
        return frame
    import cv2

    return cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_NEAREST)


def write_episode_video(
    frames,
    video_dir,
    episode_index,
    success,
    fps=10,
    resolution=512,
):
    if not frames:
        return None
    if fps <= 0:
        raise ValueError("video fps must be positive")
    import imageio.v2 as imageio

    output_dir = Path(video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "success" if success else "fail"
    output_path = output_dir / f"episode_{episode_index:03d}_{tag}.mp4"
    imageio.mimsave(
        output_path,
        [_upscale(frame, resolution) for frame in frames],
        fps=fps,
    )
    print(f"  saved {output_path}")
    return output_path
