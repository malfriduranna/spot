#!/usr/bin/env python3
"""
Convert SoccerNet Labels-v2.json annotations into the test.json format
used by this repo.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SoccerNet Labels-v2.json into data/<dataset>/test.json entries."
    )
    parser.add_argument("--label-file", required=True, help="Path to Labels-v2.json")
    parser.add_argument(
        "--video-name",
        required=True,
        help="Name to use for the video entry (must match the frame folder)",
    )
    parser.add_argument(
        "--frame-dir",
        required=True,
        help="Directory containing extracted frames for the video",
    )
    parser.add_argument(
        "--fps", type=float, default=5.0, help="Frame rate used when extracting frames"
    )
    parser.add_argument(
        "--half-duration",
        type=float,
        default=45 * 60,
        help="Duration (in seconds) assumed for each half",
    )
    parser.add_argument(
        "--class-file",
        default="data/soccernetv2/class.txt",
        help="Text file listing valid class names",
    )
    parser.add_argument(
        "--output",
        default="data/soccernetv2/test.json",
        help="Destination JSON file to overwrite",
    )
    return parser.parse_args()


def load_spotting_events(label_file):
    with open(label_file) as fp:
        data = json.load(fp)
    annotations = data.get("annotations")
    if isinstance(annotations, list):
        return annotations
    if isinstance(annotations, dict):
        return annotations.get("spotting") or annotations.get("events") or []
    return []


def game_time_to_seconds(entry):
    game_time = entry.get("gameTime")
    if not game_time:
        return float(entry.get("position", 0.0)), int(entry.get("half", 1))
    half_str, clock = [x.strip() for x in game_time.split("-", 1)]
    half = int(half_str)
    minute, second = clock.split(":")
    return int(minute) * 60 + float(second), half


def main():
    args = parse_args()
    frame_dir = Path(args.frame_dir)
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {frame_dir}")
    num_frames = len(frames)

    with open(args.class_file) as fp:
        valid_classes = {line.strip() for line in fp if line.strip()}

    events = []
    unknown_labels = set()
    spotting = load_spotting_events(args.label_file)
    for entry in spotting:
        label = entry.get("label", "Unknown").strip()
        if label not in valid_classes:
            unknown_labels.add(label)
            continue

        pos_in_half, half = game_time_to_seconds(entry)
        global_time = (half - 1) * args.half_duration + pos_in_half
        frame = int(round(global_time * args.fps))
        events.append(
            {
                "frame": frame,
                "label": label,
                "score": float(entry.get("confidence", entry.get("score", 1.0))),
            }
        )

    output_entry = {
        "video": args.video_name,
        "num_frames": num_frames,
        "fps": args.fps,
        "events": events,
    }

    Path(args.output).write_text(json.dumps([output_entry], indent=2))
    print(f"Wrote {len(events)} events to {args.output}")
    if unknown_labels:
        print("Skipped labels not in class list:", ", ".join(sorted(unknown_labels)))


if __name__ == "__main__":
    main()
