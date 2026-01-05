#!/usr/bin/env python3
"""Filter predictions and truth to a single video, sorting events by score."""

import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path) as fp:
        return json.load(fp)


def save_json(path, data):
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred-file",
        default="pretrained/soccernet_rgb/pred-test.140.json",
        help="Prediction JSON to filter",
    )
    ap.add_argument(
        "--truth-file",
        default="data/soccernetv2/test.json",
        help="Ground-truth JSON to filter",
    )
    ap.add_argument(
        "--video",
        required=True,
        help="Video name to keep (e.g., 'tott_lei_2fps')",
    )
    args = ap.parse_args()

    pred_path = Path(args.pred_file)
    truth_path = Path(args.truth_file)

    pred_data = load_json(pred_path)
    truth_data = load_json(truth_path)

    filtered_preds = [entry for entry in pred_data if entry["video"] == args.video]
    filtered_truth = [entry for entry in truth_data if entry["video"] == args.video]

    if not filtered_preds or not filtered_truth:
        raise ValueError(f"Video '{args.video}' not found in predictions or truth files.")

    for entry in filtered_preds:
        entry["events"].sort(key=lambda e: e.get("score", 0.0), reverse=True)

    save_json(pred_path, filtered_preds)
    save_json(truth_path, filtered_truth)

    print(
        f"Done. Kept {args.video}. "
        f"Sorted {len(filtered_preds[0]['events'])} events in predictions."
    )


if __name__ == "__main__":
    main()
