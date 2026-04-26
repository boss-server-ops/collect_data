from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re

# from lerobot.common.datasets.video_utils import encode_video_frames
import logging
import av
import glob
from PIL import Image
import time
import argparse


EP_RE = re.compile(r"episode_(\d+)$")


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    # vcodec: str = "h264",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    # Check encoder availability
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    # video_path.parent.mkdir(parents=True, exist_ok=overwrite)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        logging.warning(
            f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    # Get input frames
    # template = "frame_" + ("[0-9]" * 6) + ".png"
    template = "frame_" + ("[0-9]" * 6) + ".jpg"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    # Define video output frame size (assuming all input frames are the same size)
    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    dummy_image = Image.open(input_list[0])
    width, height = dummy_image.size

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_data in input_list:
            input_image = Image.open(input_data).convert("RGB")
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")
    
@dataclass(frozen=True)
class Job:
    imgs_dir: Path        # e.g. images/observation.images.right/episode_000020
    out_path: Path        # e.g. videos/observation.images.right/episode_000020.mp4
    fps: int


def episode_idx(ep_dir: Path) -> int:
    m = EP_RE.search(ep_dir.name)
    if not m:
        raise ValueError(f"Not an episode dir: {ep_dir}")
    return int(m.group(1))


def build_jobs(dataset_root: Path, fps: int) -> list[Job]:
    images_root = dataset_root / "images"
    videos_root = dataset_root / "videos/chunk-000"
    print(f"[INFO] images_root: {images_root}")
    print(f"[INFO] videos_root: {videos_root}")
    jobs: list[Job] = []
    for key_dir in sorted(images_root.iterdir()):
        print(f"[DEBUG] key_dir: {key_dir}")
        if not key_dir.is_dir():
            continue

        key = key_dir.name  # e.g. observation.images.right
        for ep_dir in sorted(key_dir.glob("episode_*")):
            print(f"[DEBUG]   ep_dir: {ep_dir}")
            if not ep_dir.is_dir():
                continue
            idx = episode_idx(ep_dir)
            print(f"[DEBUG]     episode_idx: {idx}")
            out_path = videos_root / key / f"episode_{idx:06d}.mp4"
            print(f"[DEBUG]     out_path: {out_path}")
            jobs.append(Job(imgs_dir=ep_dir, out_path=out_path, fps=fps))

    return jobs


def run_jobs(jobs: list[Job], *, num_workers: int = 2, overwrite: bool = False) -> None:
    def _run_one(job: Job) -> Path:
        job.out_path.parent.mkdir(parents=True, exist_ok=True)

        if job.out_path.exists() and not overwrite:
            return job.out_path  # skip
            
        print(f"start encoding video: {job.imgs_dir}")
        encode_video_frames(
            imgs_dir=job.imgs_dir,
            video_path=job.out_path,
            fps=job.fps,
            overwrite=overwrite,
        )
        print(f"finished encoding video: {job.out_path}")
        return job.out_path

    errors: list[tuple[Job, Exception]] = []
    done = 0

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(_run_one, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                out = fut.result()
                done += 1
                if done % 10 == 0:
                    print(f"[INFO] encoded/checked {done}/{len(jobs)} -> {out}")
            except Exception as e:
                # 记录失败但继续
                # 注意：encode_video_frames 在 imgs_dir 没有 frame_xxxxxx.png 会抛 FileNotFoundError
                errors.append((jobs[futs.index(fut)], e))

    if errors:
        print(f"[ERROR] {len(errors)} jobs failed. Showing first 10:")
        for job, e in errors[:10]:
            print(f"  - imgs_dir={job.imgs_dir} out={job.out_path} err={e}")
        raise RuntimeError(f"{len(errors)} video encodes failed")

    print(f"[INFO] all done: {done}/{len(jobs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LeRobot image sequences to videos (multi-threaded)"
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="数据集根目录，例如 /path/to/dataset"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="视频帧率 (default: 30)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="并行线程数 (default: 6)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="是否覆盖已存在的视频"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    fps = args.fps
    num_workers = args.num_workers
    overwrite = args.overwrite

    jobs = build_jobs(dataset_root, fps=fps)
    print(f"[INFO] total jobs: {len(jobs)}")

    t0 = time.perf_counter()
    run_jobs(jobs, num_workers=num_workers, overwrite=overwrite)
    print("total time:", time.perf_counter() - t0)


'''
python convert_lerobot_imgs2videos_multi_threads.py \
  --dataset_root /home/woan/MSD/lerobot_data_record/woanlerobot/dataset/x1_subtask3_v1_2026012901 \
  --fps 30 \
  --num_workers 6 
'''