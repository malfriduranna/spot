#!/usr/bin/env python3
"""Plot temporal class probabilities for SoccerNet predictions."""

import argparse
import json
import os
import pickle

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib


def get_args():
    parser = argparse.ArgumentParser(
        description='Plot per-frame probabilities for a target event.')
    parser.add_argument(
        '--frame_pkl', default='pretrained/soccernet_rgb/pred-test.140.frame.pkl',
        help='Path to the serialized frame-level predictions.')
    parser.add_argument(
        '--class_file', default='data/soccernetv2/class.txt',
        help='Path to the class label list (one label per line).')
    parser.add_argument(
        '--truth_json', default='data/soccernetv2/test.json',
        help='Ground-truth annotations to locate anchor frames.')
    parser.add_argument(
        '--video', default='arsenal_liverpool_224p_5fps',
        help='Video identifier to plot.')
    parser.add_argument(
        '--label', default='Indirect free-kick',
        help='Target class label to visualize.')
    parser.add_argument(
        '--output', default='results_figures/soccernet_rgb_indirect_fk.png',
        help='Destination for the rendered figure.')
    parser.add_argument(
        '--fps', type=float, default=5.0,
        help='Frame rate used for the SoccerNet export.')
    parser.add_argument(
        '--window', type=float, default=120.0,
        help='Temporal window (seconds) to visualize around the anchor. '
             'Set to 0 to plot the entire match.')
    parser.add_argument(
        '--smooth', type=float, default=1.0,
        help='Apply a moving-average smoother with this width (seconds). '
             'Set to 0 to disable smoothing.')
    parser.add_argument(
        '--target_frame', type=int,
        help='Optional ground-truth frame to lock onto. '
             'When omitted, the event with the highest response is used.')
    parser.add_argument(
        '--search_radius', type=float, default=10.0,
        help='Radius (seconds) to search around each ground-truth anchor when '
             'finding the strongest response.')
    return parser.parse_args()


def load_frame_scores(pred_path, video, label, class_file):
    with open(class_file, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    if label not in classes:
        raise ValueError(f'Label "{label}" not found in {class_file}')

    with open(pred_path, 'rb') as f:
        frame_scores = pickle.load(f)
    if video not in frame_scores:
        raise KeyError(f'Video "{video}" missing from {pred_path}')

    class_idx = classes.index(label)
    # frame_scores[video] is a tuple (logits, background); select logits.
    return frame_scores[video][0][:, class_idx]


def load_truth_annotations(truth_path, video, label):
    with open(truth_path, 'r') as f:
        truth = json.load(f)
    for entry in truth:
        if entry['video'] == video:
            return [e['frame'] for e in entry['events'] if e['label'] == label]
    raise KeyError(f'Video "{video}" missing from {truth_path}')


def pick_anchor(events, scores, fps, search_radius, target_frame=None):
    if not events:
        raise ValueError('No ground-truth events for the requested label.')
    if target_frame is not None:
        anchor = min(events, key=lambda f: abs(f - target_frame))
        window_peak = target_frame
        peak_val = scores[target_frame]
    else:
        search_frames = int(round(search_radius * fps))
        peak_val = float('-inf')
        anchor = events[0]
        window_peak = events[0]
        for frame in events:
            start = max(frame - search_frames, 0)
            end = min(frame + search_frames + 1, len(scores))
            window = scores[start:end]
            idx = np.argmax(window)
            if window[idx] > peak_val:
                peak_val = float(window[idx])
                anchor = frame
                window_peak = start + idx
    return anchor, window_peak, peak_val


def moving_average(signal, width_frames):
    if width_frames <= 1:
        return signal
    kernel = np.ones(width_frames, dtype=np.float32) / width_frames
    return np.convolve(signal, kernel, mode='same')


def plot_probabilities(scores, fps, anchor_frame, peak_frame, output_path,
                       smooth_width, window_seconds, label):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_frames = len(scores)
    if window_seconds and window_seconds > 0:
        half_window = int(round(window_seconds * fps / 2))
        start = max(anchor_frame - half_window, 0)
        end = min(anchor_frame + half_window, total_frames)
    else:
        start, end = 0, total_frames

    x = np.arange(start, end)
    y = scores[start:end]
    if smooth_width and smooth_width > 0:
        y = moving_average(y, max(1, int(round(smooth_width * fps))))

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: E402

    plt.figure(figsize=(10, 3))
    plt.plot(x / fps / 60.0, y, label=f'{label} response')
    plt.axvline(anchor_frame / fps / 60.0, color='tab:red',
                linestyle='--', label='Ground-truth anchor')
    plt.scatter([peak_frame / fps / 60.0], [scores[peak_frame]],
                color='black', zorder=5, label='Model peak')
    plt.xlabel('Match time (minutes)')
    plt.ylabel('Score')
    plt.title('Temporal probability curve')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = get_args()

    scores = load_frame_scores(
        args.frame_pkl, args.video, args.label, args.class_file)
    events = load_truth_annotations(
        args.truth_json, args.video, args.label)
    anchor_frame, peak_frame, _ = pick_anchor(
        events, scores, args.fps, args.search_radius, args.target_frame)
    plot_probabilities(
        scores, args.fps, anchor_frame, peak_frame,
        args.output, args.smooth, args.window, args.label)


if __name__ == '__main__':
    main()
