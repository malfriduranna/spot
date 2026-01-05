import argparse
import subprocess
from pathlib import Path

def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def extract_frames(video_path: Path, out_dir: Path, fps: int, height: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ffmpeg filter: fps + scale (keep aspect ratio). width=-2 keeps divisible by 2.
    vf = f"fps={fps},scale=-2:{height}"

    # Output pattern
    out_pattern = str(out_dir / "%06d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-q:v", "2",              # good jpeg quality
        out_pattern
    ]
    run(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", type=str, required=True, help="Folder with .mp4/.mkv videos")
    ap.add_argument("--frames_dir", type=str, required=True, help="Output frames root folder")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--height", type=int, default=224)
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted([
        p for p in videos_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi"}
    ])
    if not video_files:
        raise FileNotFoundError(f"No videos found in {videos_dir}")

    for vp in video_files:
        video_id = vp.stem
        out = frames_dir / video_id
        print(f"\n==> Extracting {vp.name} -> {out}")
        extract_frames(vp, out, args.fps, args.height)

if __name__ == "__main__":
    main()
