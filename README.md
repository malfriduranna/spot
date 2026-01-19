# Spotting Temporally Precise, Fine-Grained Events in Video

This repository is used for Malfridur Anna Eiriksdottir's thesis experiments while keeping the original ECCV 2022 implementation intact. The only changes from the upstream repo are (1) a minimal `data/myvideos` split that was used to verify the pipeline on personal recordings and (2) the `results_figures/spotting_spike.png` visualization referenced in the thesis. The upstream codebase and authorship remain fully attributed to the Spotting Temporally Precise, Fine-Grained Events in Video paper described below.

This repository contains code for our paper:

*Spotting Temporally Precise, Fine-Grained Events in Video*\
In ECCV 2022\
James Hong, Haotian Zhang, Michael Gharbi, Matthew Fisher, Kayvon Fatahalian

Link: [project website](https://jhong93.github.io/projects/spot.html)

```
@inproceedings{precisespotting_eccv22,
    author={Hong, James and Zhang, Haotian and Gharbi, Micha\"{e}l and Fisher, Matthew and Fatahalian, Kayvon},
    title={Spotting Temporally Precise, Fine-Grained Events in Video},
    booktitle={ECCV},
    year={2022}
}
```

This code is released under the BSD-3 [LICENSE](/LICENSE).

## Overview

Our paper presents a study of temporal event detection (spotting) in video at the precision of a single or small (e.g., 1-2) tolerance of frames.
This is a useful task for annotating video for analysis and synthesis when temporal precision is important, and we demonstrate fine-grained events in several sports as examples.

In this regime, the most crucial design aspect is end-to-end learning of spatial-temporal features from the pixels.
We present a surprisingly strong, compact, and end-to-end learned baseline that is conceptually simpler than the two-phase architectures common in the temporal action detection, segmentation, and spotting literature.

## Thesis context

This fork keeps the original ECCV 2022 implementation intact while adding a few lightweight assets that support Malfridur Anna Eiriksdottir's thesis work:

- `data/myvideos/` holds a minimal dataset configuration (currently just `class.txt` and `test.json`) used to verify that the end-to-end training and inference pipeline works on personal recordings. Replace these files with your own split definitions to spot new classes or practice sessions.
- `results_figures/spotting_spike.png` is an exported visualization of a predicted event that is referenced in the written thesis.

Everything else remains consistent with the upstream repository so the paper implementation can still be reproduced verbatim.

## Environment

The code is tested in Linux (Ubuntu 16.04 and 20.04) with the dependency versions in ```requirements.txt```. Similar environments are likely to work also but YMMV.

## Datasets

Refer to the READMEs in the [data](/data) directory for pre-processing and setup instructions.

## Basic usage

To train a model, use `python3 train_e2e.py <dataset_name> <frame_dir> -s <save_dir> -m <model_arch>`.

* `<dataset_name>`: supports tennis, fs_comp, fs_perf, finediving, finegym, soccernetv2, soccernet_ball
* `<frame_dir>`: path to the extracted frames
* `<save_dir>`: path to save logs, checkpoints, and inference results
* `<model_arch>`: feature extractor architecture (e.g., RegNet-Y 200MF w/ GSM : `rny002_gsm`)

Training will produce checkpoints, predictions for the `val` split, and predictions for the `test` split on the best validation epoch.

To evaluate a set of predictions with the mean-AP metric, use `python3 eval.py -s <split> <model_dir_or_prediction_file>`.
* `<model_dir_or_prediction_file>`: can be the saved directory of a model containing predictions or a path to a prediction file.

The predictions are saved as either `pred-{split}.{epoch}.recall.json.gz` or `pred-{split}.{epoch}.json` files. The latter contains only the top class predictions for each frame, omitting all background, while the former contains all non-background detections above a low threshold, to complete the precision-recall curve.

We also save per-frame scores, `pred-{split}.{epoch}.score.json.gz`, which can be used to combine predictions from multiple models (see `eval_ensemble.py`).

### Trained models

Models and configurations can be found at https://github.com/jhong93/e2e-spot-models/. Place the checkpoint file and config.json file in the same directory.

To perform inference with an already trained model, use `python3 test_e2e.py <model_dir> <frame_dir> -s <split> --save`. This will save the predictions in the model directory, using the default file naming scheme.

### Baselines

Implementations for several baselines in the paper are in `baseline,py`. TSP and 2D-VPD features are available in our [Google Drive](https://drive.google.com/drive/folders/1AQFd8JsvxdEG2jQfY5GDVSLEtc9r824W?usp=sharing).

## Using your own data

Each dataset has plaintext files that contain the list of classes and events in each video.

#### class.txt

This is a list of the class names, one per line.

#### {split}.json

This file contains entries for each video and its contained events.

```
[
    {
        "video": VIDEO_ID,
        "num_frames": 4325,                 // Video length
        "num_events": 10,
        "events": [
            {
                "frame": 525,               // Frame
                "label": CLASS_NAME,        // Event class
                "comment": ""               // Optional comments
            },
            ...
        ],
        "fps": 25,
        "width": 1920,      // Metadata about the source video
        "height": 1080
    },
    ...
]
```

#### Frame directory

We assume pre-extracted frames (either RGB in jpg format or optical flow), that have been resized to 224 pixels high or similar. The organization of the frames is expected to be `<frame_dir>/<video_id>/<frame_number>.jpg`. For example,

```
video1/
├─ 000000.jpg
├─ 000001.jpg
├─ 000002.jpg
├─ ...
video2/
├─ 000000.jpg
├─ ...
```

#### Prediction file format

Predictions are formatted similarly to the labels:
```
[
    {
        "video": VIDEO_ID,
        "events": [
            {
                "frame": 525,               // Frame
                "label": CLASS_NAME,        // Event class
                "score": 0.96142578125
            },
            ...
        ],
        "fps": 25           // Metadata about the source video
    },
    ...
]
```

## Thesis helper scripts and SoccerNet testing

All thesis-specific utilities live in [scripts/](scripts):

- `download_test_video.py` grabs a single match from the SoccerNet NDA bucket (supply your own credentials) so the rest of the pipeline can run without fetching the entire dataset.
- `convert_soccernet_labels.py` rewrites the SoccerNet `Labels-v2.json` annotations for a downloaded match into the `data/<dataset>/<split>.json` format expected by this repo, including clock-to-frame conversion at the extraction FPS.
- `convert_frame_predictions.py` turns the raw `pred-*.frame.pkl` files dumped by `test_e2e.py` into the JSON / `.recall.json.gz` / `.score.json.gz` artifacts used for evaluation and visualization.
- `filter_single_video.py` trims both predictions and ground truth to a single video to keep notebooks and plots lightweight.
- `plot_temporal_probability.py` and `batch_plot_soccernet.py` visualize per-frame logits over time (optionally smoothed) and mark both the ground-truth anchors and the highest-confidence detections, which is helpful when copying figures into the thesis.
- `cluster/test_e2e_gpu100.bsub` is the DTU compute-cluster submission script used to execute inference runs on an A100 node: it requests 4 CPU cores, 16 GB RAM, one GPU, and a 4‑hour wall clock; loads `python3/3.9.19` and `cuda/12.1`; activates the `~/.virtualenvs/spot` environment; changes into the submission directory; ensures `logs/` exists; and finally runs `python3 test_e2e.py pretrained/soccernet_rgb frames/short_5fps -s test --save -d soccernetv2` while teeing stdout/stderr into `%J.out/.err` so each job’s output can be inspected afterwards.

### How the SoccerNet evaluation was reproduced

1. Downloaded an EPL match with `scripts/download_test_video.py`, extracted frames at 5 FPS to `frames/full_match_5fps/` (using the existing `frames_as_jpg_soccernet.py` helper), and converted its official annotations via `scripts/convert_soccernet_labels.py --video-name <video_id> --frame-dir frames/full_match_5fps --label-file <path-to-Labels-v2.json> --fps 5`.
2. Ran inference with the released checkpoint in `pretrained/soccernet_rgb` (`python3 test_e2e.py pretrained/soccernet_rgb frames/full_match_5fps -s test --save`) to generate `pred-test.140.frame.pkl`.
3. Converted those frame-level logits into event JSON/GZ files via `scripts/convert_frame_predictions.py --model-dir pretrained/soccernet_rgb --prefix pred-test.140 --dataset soccernetv2 --frame-dir frames/full_match_5fps --split test`.
4. Used `scripts/filter_single_video.py --video <video_id>` as needed to isolate the match for qualitative review, and produced the plots that appear in the thesis with `scripts/plot_temporal_probability.py` (or the batch variant for multiple anchors).

These steps verified that the upstream model runs end-to-end on SoccerNet data inside this fork, and the resulting files are what you see under `pretrained/soccernet_rgb/` and `results_figures/`.
