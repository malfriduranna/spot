#!/usr/bin/env python3
"""Batch-generate temporal probability plots for SoccerNet videos."""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(
        description='Generate multiple temporal probability plots.')
    parser.add_argument('--video', required=True,
                        help='Video identifier from the SoccerNet dataset.')
    parser.add_argument('--frame_pkl', required=True,
                        help='Path to frame-level logits (pickle).')
    parser.add_argument('--class_file', default='data/soccernetv2/class.txt',
                        help='Path to the class label list.')
    parser.add_argument('--truth_json', default='data/soccernetv2/test.json',
                        help='Ground-truth annotations with event frames.')
    parser.add_argument('--fps', type=float, required=True,
                        help='Frame rate of the extracted video.')
    parser.add_argument('--window', type=float, default=120.0,
                        help='Temporal window (seconds) for each plot.')
    parser.add_argument('--smooth', type=float, default=1.0,
                        help='Smoothing window (seconds) for probability curves.')
    parser.add_argument('--max_per_label', type=int, default=3,
                        help='Number of anchors plotted per label (0 for all).')
    parser.add_argument('--labels', nargs='*',
                        help='Optional subset of labels to plot. '
                             'Defaults to all labels present in the video.')
    parser.add_argument('--output_dir', default='results_figures/batch_plots',
                        help='Directory to store generated figures.')
    return parser.parse_args()


def load_events(truth_path, video):
    with open(truth_path, 'r') as f:
        data = json.load(f)
    for entry in data:
        if entry['video'] == video:
            events = defaultdict(list)
            for e in entry['events']:
                events[e['label']].append(e['frame'])
            return events
    raise KeyError(f'Video "{video}" not found in {truth_path}')


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def main():
    args = get_args()
    events_by_label = load_events(args.truth_json, args.video)

    target_labels = sorted(events_by_label.keys())
    if args.labels:
        missing = set(args.labels) - set(events_by_label.keys())
        if missing:
            raise ValueError(
                f'Labels {missing} not found for video {args.video}')
        target_labels = args.labels

    output_dir = ensure_output_dir(args.output_dir)
    script_path = os.path.join(os.path.dirname(__file__),
                               'plot_temporal_probability.py')

    for label in target_labels:
        frames = sorted(events_by_label[label])
        if args.max_per_label > 0:
            frames = frames[:args.max_per_label]

        for frame in frames:
            cmd = [
                sys.executable, script_path,
                '--frame_pkl', args.frame_pkl,
                '--class_file', args.class_file,
                '--truth_json', args.truth_json,
                '--video', args.video,
                '--label', label,
                '--fps', str(args.fps),
                '--window', str(args.window),
                '--smooth', str(args.smooth),
                '--target_frame', str(frame),
                '--output', output_dir
            ]
            print('Running:', ' '.join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
