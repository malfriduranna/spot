#!/usr/bin/env python3
"""Convert frame-level prediction pickles into JSON/GZ event files."""

import argparse
import numpy as np
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.frame import ActionSpotVideoDataset
from util.dataset import load_classes
from util.eval import process_frame_predictions
from util.io import load_json, store_json, store_gz_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert pred-*.frame.pkl files into JSON/GZ predictions."
    )
    parser.add_argument(
        "--model-dir",
        default="pretrained/soccernet_rgb",
        help="Directory containing config.json and pred-*.frame.pkl",
    )
    parser.add_argument(
        "--prefix",
        default="pred-test.140",
        help="Prediction file prefix (e.g., pred-test.140)",
    )
    parser.add_argument(
        "--dataset",
        default="soccernetv2",
        help="Dataset name (used to locate data/<dataset>/class.txt)",
    )
    parser.add_argument(
        "--frame-dir",
        default="frames",
        help="Root directory containing extracted frames",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to use (train/val/test/challenge)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json(f"{args.model_dir}/config.json")
    classes = load_classes(f"data/{args.dataset}/class.txt")

    split_data = ActionSpotVideoDataset(
        classes,
        f"data/{args.dataset}/{args.split}.json",
        args.frame_dir,
        config["modality"],
        config["clip_len"],
        overlap_len=0,
        crop_dim=config["crop_dim"],
        skip_partial_end=False,
    )

    with open(f"{args.model_dir}/{args.prefix}.frame.pkl", "rb") as fp:
        pred_dict = pickle.load(fp)

    # Normalize scores and ensure support > 0 everywhere
    for video, (scores, support) in pred_dict.items():
        scores = np.asarray(scores, dtype=np.float32)
        support = np.asarray(support, dtype=np.float32)
        support[support <= 0] = 1.0
        scores /= support[:, None]
        pred_dict[video] = (scores, np.ones_like(support, dtype=np.int32))

    _, _, events, events_hr, frame_scores = process_frame_predictions(
        split_data, classes, pred_dict
    )

    store_json(f"{args.model_dir}/{args.prefix}.json", events)
    store_gz_json(f"{args.model_dir}/{args.prefix}.recall.json.gz", events_hr)
    store_gz_json(f"{args.model_dir}/{args.prefix}.score.json.gz", frame_scores)
    print(f"Wrote JSON/GZ predictions for {args.prefix}")


if __name__ == "__main__":
    main()
