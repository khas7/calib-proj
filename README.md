<div align="center">

<h1>
CalibProj: Automatic Calibration of a Multi-Camera System  
with Limited Overlapping Fields of View  
for 3D Surgical Scene Reconstruction
</h1>

[**Tim Flückiger**](https://www.linkedin.com/in/timflueckiger/)<sup>★</sup> ·
[**Jonas Hein**](https://scholar.google.com/citations?user=Kk_o9AYAAAAJ&hl=fr&oi=ao) ·
[**Valery Fischer**](https://www.linkedin.com/in/valery-fischer/) <br>
[**Philipp Fürnstahl**](https://scholar.google.com/citations?user=nQ4B3BgAAAAJ&hl=fr) ·
[**Lilian Calvet**](https://scholar.google.com/citations?user=6JewdrMAAAAJ&hl=fr)

<span>★</span> corresponding author

<a href="https://doi.org/10.1007/s11548-025-03413-1">
  <img src="https://img.shields.io/badge/paper-grey" alt="Paper PDF" height="30">
</a>

<a href="https://tflueckiger.github.io/calib-proj/">
  <img src="https://img.shields.io/badge/Project_Page-CalibProj-blue" alt="Project Page" height="30">
</a>

</div>

## Automatic Multi-Camera External Calibration Using Projected Markers

CalibProj is a Python toolkit for automatically estimating the extrinsic parameters of multi-camera systems with limited field-of-view overlap.

The original method uses multi-scale markers projected onto a planar surface. This extension instead processes markers of a **single predefined size** projected onto the non-planar surface of an operating table.

The target cameras are near-field cameras observing the surgical field. The projected markers should therefore be visible on the operating table from the relevant near-field viewpoints.

The projector does not need to be geometrically calibrated.

## Installation

### Install `calib-commons`

```bash
git clone https://github.com/tflueckiger/calib-commons.git
cd calib-commons

pip install .

# Optional editable installation
pip install -e . --config-settings editable_mode=strict
```

### Install CalibProj

```bash
git clone https://github.com/tflueckiger/calib-proj.git
cd calib-proj

pip install .

# Optional editable installation
pip install -e . --config-settings editable_mode=strict
```

## Prerequisites

Before running the calibration:

1. The intrinsic parameters of all cameras must be known.
2. The camera videos must already be temporally synchronized using [ROCSync](https://github.com/jaromeyer/RocSync).

The filenames of the synchronized videos and intrinsic calibration files must use matching camera identifiers.

```text
videos/
├── camera02.mp4
├── camera03.mp4
└── ...

intrinsics/
├── camera02_intrinsics.json
├── camera03_intrinsics.json
└── ...
```

## How to Use

### 1. Generate a Projection Sequence

#### Moving Markers

```bash
python .\scripts\generate_moving_markers.py -h
```

```text
usage: generate_moving_markers.py [-h] --out_file OUT_FILE
                                  [--show_image]
                                  [--projector_height PROJECTOR_HEIGHT]
                                  [--effective_size EFFECTIVE_SIZE]
                                  [--color_space]
                                  [--marker_sizes MARKER_SIZES]
```

| Argument             | Description                                                                         |
| -------------------- | ----------------------------------------------------------------------------------- |
| `--out_file`         | Output path of the generated video                                                  |
| `--show_image`       | Displays the first generated image for debugging                                    |
| `--projector_height` | Projector height above the projection surface; default: `1.1 m`                     |
| `--effective_size`   | Effective projected area in metres `[y, x]`; default: `[0.4, 2.0]`                  |
| `--color_space`      | Projects black pixels in the red color channel to improve contrast on some surfaces |
| `--marker_sizes`     | Marker sizes to generate; default: `all`                                            |

Example:

```bash
python .\scripts\generate_moving_markers.py \
    --out_file video\moving_markers.mp4 \
    --marker_sizes 135
```

#### Static Markers

```bash
python .\scripts\generate_static_markers.py -h
```

```text
usage: generate_static_markers.py [-h] --out_file OUT_FILE
                                  [--show_image]
                                  [--color_space]
                                  [--marker_size MARKER_SIZE]
```

| Argument        | Description                                    |
| --------------- | ---------------------------------------------- |
| `--out_file`    | Output path of the generated image or sequence |
| `--show_image`  | Displays the generated image for debugging     |
| `--color_space` | Displays black pixels in the red color channel |
| `--marker_size` | Marker size to generate; default: `all`        |

Example:

```bash
python .\scripts\generate_static_markers.py \
    --out_file video\static_marker.png \
    --marker_size 135
```

### 2. Data Acquisition

Mount the projector on the ceiling and align the projection with the operating table.

The recommended acquisition sequence is:

1. Start all cameras.
2. Record ROCSync.
3. Record the checkerboard for reference calibration.
4. Project the marker sequence.
5. Record the checkerboard again.
6. Record ROCSync again.

The videos must subsequently be synchronized using ROCSync before running the calibration.

### 3. Run the Calibration

```bash
python .\scripts\run.py -h
```

```text
usage: run.py [-h]
              --video_folder VIDEO_FOLDER
              --intrinsics_folder INTRINSICS_FOLDER
              [--output_folder OUTPUT_FOLDER]
              --sequence_info_path SEQUENCE_INFO_PATH
              --start_time START_TIME
              --end_time END_TIME
              [--alpha ALPHA]
              [--beta BETA]
              [--gamma GAMMA]
              [--clahe_clip CLAHE_CLIP]
              [--clahe_grid CLAHE_GRID]
              [--debug_preprocessing]
              [--save_scene]
```

Required arguments:

| Argument               | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `--video_folder`       | Folder containing the synchronized camera videos |
| `--intrinsics_folder`  | Folder containing the camera intrinsics          |
| `--sequence_info_path` | Path to the projection-sequence metadata         |
| `--start_time`         | Exact start time of the marker sequence          |
                         | 'HH:MM:SS.ms'                                    |
| `--end_time`           | Exact end time of the marker sequence            |
                         | 'HH:MM:SS.ms'                                    |

Optional arguments:

| Argument                | Description                                         |
| ----------------------- | --------------------------------------------------- |
| `--output_folder`       | Parent output folder; default: `results`            |
| `--alpha`, `--beta`     | Lower and upper bounds for image normalization      |
| `--gamma`               | Gamma-correction value                              |
| `--clahe_clip`          | CLAHE clip limit                                    |
| `--clahe_grid`          | CLAHE grid size                                     |
| `--debug_preprocessing` | Displays preprocessing and marker-detection results |
| `--save_scene`          | Saves the reconstructed scene as a pickle file      |

Example:

```bash
python .\scripts\run.py \
    --video_folder <path-to-video-folder> \
    --intrinsics_folder <path-to-camera-intrinsics> \
    --sequence_info_path <path-to-sequence-info> \
    --start_time <start-time> \
    --end_time <end-time> \
    --output_folder results
```

The input folders must follow this structure:

```text
<path-to-video-folder>/
├── camera02.mp4
├── camera03.mp4
└── ...

<path-to-camera-intrinsics>/
├── camera02_intrinsics.json
├── camera03_intrinsics.json
└── ...
```

## Output

The estimated camera poses are stored in:

```text
results/camera_poses.json
```

The calibration metrics are stored in:

```text
results/metrics.json
```

The metrics include:

* mean reprojection error,
* standard deviation of the reprojection error,
* camera view score,
* number of valid correspondences.

## Evaluate the Camera Poses

The estimated camera poses can be compared with a reference calibration using:

```bash
python .\scripts\eval_calib.py -h
```

```text
usage: eval_calib.py [-h]
                     --path PATH
                     --gt_path GT_PATH
                     [--scale]
                     [--save_json]
```

| Argument      | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| `--path`      | Path to the estimated camera-pose file                          |
| `--gt_path`   | Path to the reference-calibration file                          |
| `--scale`     | Estimates an optimal scale when the translation unit is unknown |
| `--save_json` | Saves the aligned poses, scale factor, and pose errors          |

Example:

```bash
python .\scripts\eval_calib.py \
    --path results\camera_poses.json \
    --gt_path <path-to-reference-calibration> \
    --scale \
    --save_json
```

## Citation

```bibtex
@article{fluckiger2025,
  author  = {Tim Flückiger and Jonas Hein and Valery Fischer and Philipp Fürnstahl and Lilian Calvet},
  title   = {Automatic calibration of a multi-camera system with limited overlapping fields of view for 3D surgical scene reconstruction},
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  year    = {2025},
  doi     = {10.1007/s11548-025-03413-1},
  url     = {https://doi.org/10.1007/s11548-025-03413-1}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/tflueckiger/calib-proj/blob/main/LICENSE) file for details.

## Acknowledgments

This work has been supported by [OR-X](https://or-x.ch/en/translational-center-for-surgery/), a Swiss national research infrastructure for translational surgery, and by the University of Zurich and University Hospital Balgrist.
