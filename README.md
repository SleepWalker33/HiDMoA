# HiDMoA

HiDMoA is a class-incremental defect classification method based on task-wise MoE
adapters, prototype heads, and a feature-VAE task router.

This repository contains the extracted HiDMoA method only. Experiment output
folders, ablation launchers, and unrelated baseline repositories are not required
for normal use.

The public method name is `HiDMoA`. In the original research code the same method
was named `incremental_2`; the config key is still `INCREMENTAL_2` for backward
compatibility.

## 1. What to run first

```bash
cd /userhome/home/yangdandan/ydd/Incre_class_project/Incre_neu/github
conda env create -f configs/environment-mmlab.yml   # first time only
conda activate mmlab
pip install -e .
hidmoa --help
hidmoa-check-data --help
hidmoa-prepare-folders --help
```

If this is a fresh machine, create and activate `mmlab` before `pip install -e .`, otherwise keep your existing Python environment.

## 2. Data format and class encoding

- `labels/*.txt` first token as class id (numeric) is preferred.
- Label files with non-numeric class names are also supported.
- Filename prefix fallback is supported when label files are unavailable.

## 3. Reproducing incremental_2 (HiDMoA)

```bash
hidmoa --dataset neu_xsdd --seed 42 --repeats 1 --device cuda
```

`hidmoa` maps to incremental_2 by default.

Or use the script entry directly (equivalent after installation and in the same
environment):

```bash
python examples/run_hidmoa.py --dataset neu_xsdd --seed 42 --repeats 1 --device cuda
```

Both commands are acceptable; use `hidmoa` when the package is installed, and
use `python ...` when you only want to run the module directly.

## 4. Main tuning knobs

- Priority parameters (`*` means key parameters):
  1*. `HIDMOA_BATCH_SIZE`
  2*. `HIDMOA_EPOCHS_PER_TASK`
  3*. `train.early_stopping_patience` (`HIDMOA_EARLY_STOP_PATIENCE`)
  4*. `train.lr`
  5*. `train.weight_decay`
  6*. `model.backbone`
  7*. `model.experts_per_task`

- All `INCREMENTAL_2` knobs are in `src/hidmoa/config.py` (`model.*`, `train.*`, `vae/router` fields); full prioritized list is in `Important Hyperparameters`.
- For quick smoke tests, use `--quick-test`.

## Repository Layout

```text
.
  README.md
  pyproject.toml
  configs/
    hidmoa_env_example.sh
    environment-mmlab.yml
    requirements-mmlab.txt
  examples/
    run_hidmoa.py
  src/hidmoa/
    cli.py
    config.py
    data.py
    main.py
    models.py
    train.py
```

## Install

Create the exported MMLab environment first. This is the recommended path for a
fresh machine or a new user.

```text
configs/environment-mmlab.yml
configs/requirements-mmlab.txt
```

Run:

```bash
conda env create -f configs/environment-mmlab.yml
conda activate mmlab
cd /path/to/hidmoa
pip install -e .
```

`environment-mmlab.yml` is the preferred installation file. `requirements-mmlab.txt`
is a flat `pip list --format=freeze` record of the same environment and is useful
for auditing exact package versions. If a fresh install fails on CUDA/MMCV wheels,
install PyTorch/MMCV according to your CUDA driver first, then run `pip install -e .`.

The exported environment was based on Python 3.8 and includes the important runtime
packages used during development, including:

```text
torch==2.4.1
torchvision==0.19.1
mmcv==2.1.0
mmengine==0.10.6
mmdet==3.3.0
matplotlib
numpy
scikit-learn
thop
timm
optuna
```

After installation, these commands should be available:

```bash
hidmoa --help
hidmoa-check-data --help
hidmoa-prepare-folders --help
```

They are used as follows:

```text
hidmoa                 train and evaluate HiDMoA
hidmoa-check-data      check dataset paths, labels, class counts, and task splits before training
hidmoa-prepare-folders convert raw class folders into the required images/ and labels/ layout
```

The `--help` flag only prints command-line usage. It is useful for verifying that
installation succeeded and for checking available arguments.

## Dataset Format

Each dataset root should use this classification-friendly YOLO-style layout:

```text
dataset_root/
  classes.txt        # recommended metadata; not required by the default built-in presets
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Training uses the class names and incremental task order defined in
`src/hidmoa/config.py` (`DATASET_PRESETS`). For the built-in NEU/XSDD presets,
`classes.txt` is not the source of truth. It is still recommended because it
documents the dataset, is produced by `hidmoa-prepare-folders`, and can be used by
auto-loaded/custom dataset presets.

`classes.txt`, when present, should use indexed names:

```text
0 crazing
1 inclusion
2 patches
```

Each image should have a label file with the same stem:

```text
images/train/crazing_000001.jpg
labels/train/crazing_000001.txt
```

For classification, the simplest label file contains only the class id:

```text
0
```

Example:

```text
labels/train/crazing_000001.txt      -> 0
labels/train/inclusion_000001.txt    -> 1
labels/val/patches_000001.txt        -> 2
labels/test/crazing_000002.txt       -> 0
```

HiDMoA reads only the first token in the label file as the class id. Existing
YOLO detection labels are also accepted:

```text
0 0.2 0.35 0.92 0.89
```

This means:

```text
0     class id
0.2   bounding-box center x, normalized to [0, 1]
0.35   bounding-box center y, normalized to [0, 1]
0.92   bounding-box width, normalized to [0, 1]
0.89   bounding-box height, normalized to [0, 1]
```

For this classification code, only `0` is used. The four box values can be
placeholders and are ignored. If `labels/` is absent, HiDMoA falls back to
filename prefixes such as `crazing_000001.jpg`, but label files are recommended.

## Convert Raw Class Folders

If your raw data is organized as:

```text
raw_root/
  crazing/
    a.jpg
    b.jpg
  inclusion/
    c.jpg
```

convert it to the HiDMoA layout:

```bash
hidmoa-prepare-folders \
  --raw-root /data/raw_defects \
  --out-root /data/defects_yolo_cls \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

This creates `images/train`, `images/val`, `images/test`, matching `labels/*`,
and `classes.txt`.

## Configure Dataset Paths

The repository already includes the preprocessed NEU and XSDD example datasets:

```text
data/neudata_yolo_701515
data/xsdd_yolo_cls_701515
```

For the default `neu_xsdd` experiment, no dataset path variables are required.
You can directly run:

```bash
hidmoa-check-data --dataset neu_xsdd
hidmoa --dataset neu_xsdd --seed 42 --repeats 1 --device cuda
```

If you want to use your own copies of the datasets, override the paths with
environment variables:

```bash
export CIL_ACTIVE_DATASET=neu_xsdd
export CIL_NEU_ROOT=/data/neudata_yolo_701515
export CIL_XSDD_ROOT=/data/xsdd_yolo_cls_701515
export CIL_SEED=42
export CIL_REPEATS=1
```

You can also copy and edit:

```bash
source configs/hidmoa_env_example.sh
```

Common built-in dataset path variables:

```text
CIL_NEU_ROOT
CIL_XSDD_ROOT
CIL_MAGNETIC_ROOT
CIL_CR7_ROOT
CIL_GC10_ROOT
CIL_DAGM_ROOT
CIL_KOLEK_ROOT
CIL_BSD_ROOT
```

Subject-data presets use variables such as `CIL_SUBJECT_CIFAR10_ROOT`,
`CIL_SUBJECT_EUROSAT_ROOT`, and `CIL_SUBJECT_PATTERNNET_ROOT`.

## Define Your Own Task Name And Task Split

Dataset/task names are defined in `src/hidmoa/config.py`.

### Single Dataset Example

For one custom dataset, add a data root, class list, and preset:

```python
DATA_ROOTS["mydata"] = os.getenv("CIL_MYDATA_ROOT", "/data/mydata_yolo_cls")

MYDATA_CLASSES = [
    "class_a",
    "class_b",
    "class_c",
    "class_d",
    "class_e",
    "class_f",
]

DATASET_PRESETS["mydata"] = _preset(
    MYDATA_CLASSES,
    [[0, 1, 2], [3, 4, 5]],
)
```

Then run:

```bash
export CIL_ACTIVE_DATASET=mydata
export CIL_MYDATA_ROOT=/data/mydata_yolo_cls
hidmoa-check-data --dataset mydata
hidmoa --dataset mydata --seed 42 --repeats 1 --device cuda
```

The task split list controls the class-incremental sessions. For example,
`[[0, 1, 2], [3, 4, 5]]` means:

```text
session 0: class 0, class 1, class 2
session 1: class 3, class 4, class 5
```

When splitting one dataset into multiple tasks, group classes so that classes
within the same task are as similar as possible, while classes from different
tasks are as different as possible. This makes the incremental sessions more
semantically separated and easier to interpret.

### NEU + XSDD Example

The default example used in this repository is `neu_xsdd`. It is already defined
in `src/hidmoa/config.py`, and the preprocessed data is included under `data/`.
You can run it directly:

```bash
hidmoa-check-data --dataset neu_xsdd
hidmoa --dataset neu_xsdd --seed 42 --repeats 1 --device cuda
```

The corresponding config pattern is:

```python
DATA_ROOTS["neu"] = os.getenv("CIL_NEU_ROOT", _repo_data_path("neudata_yolo_701515"))
DATA_ROOTS["xsdd"] = os.getenv("CIL_XSDD_ROOT", _repo_data_path("xsdd_yolo_cls_701515"))

NEU_CLASSES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled-in_Scale",
    "Scratches",
]

XSDD_CLASSES = [
    "finishing_roll_printing",
    "iron_sheet_ash",
    "oxide_scale_of_plate_system",
    "oxide_scale_of_temperature_system",
    "red_iron",
    "slag_inclusion",
    "surface_scratch",
]

BASE_DATASET_PRESETS["neu"] = _preset(NEU_CLASSES, [range(len(NEU_CLASSES))])
BASE_DATASET_PRESETS["xsdd"] = _preset(XSDD_CLASSES, [range(len(XSDD_CLASSES))])

DATASET_PRESETS["neu_xsdd"] = _compose_dataset_chain("neu", "xsdd")
```

This creates two incremental sessions. Each dataset is treated as one task:

```text
session 0 / task 1: all NEU classes, global class ids 0-5
session 1 / task 2: all XSDD classes, global class ids 6-12
```

NEU has 6 classes, so XSDD class ids are offset by 6 in the combined task. If
you add more task datasets later, append them to `_compose_dataset_chain`, for
example `_compose_dataset_chain("neu", "xsdd", "magnetic", "cr7")`; each added
dataset becomes the next incremental session.

If you use different local paths, override them before checking data and training:

```bash
export CIL_NEU_ROOT=/data/neudata_yolo_701515
export CIL_XSDD_ROOT=/data/xsdd_yolo_cls_701515

hidmoa-check-data --dataset neu_xsdd
hidmoa --dataset neu_xsdd --seed 42 --repeats 1 --device cuda
```

## Check Data Before Training

Run this before training:

```bash
hidmoa-check-data --dataset neu_xsdd
```

It verifies:

- dataset roots exist
- `images/train`, `images/val`, `images/test` exist
- labels or filename prefixes can be parsed
- each split can be loaded
- class counts and task splits are visible

## Train

Quick smoke test:

`--quick-test` uses the bundled NEU/XSDD data, 1 epoch per stage, and a small
per-class sample subset so that the full pipeline can be checked quickly.

```bash
hidmoa \
  --dataset neu_xsdd \
  --seed 42 \
  --repeats 1 \
  --device cuda \
  --quick-test
```

CPU-only smoke test:

```bash
hidmoa --dataset neu_xsdd --seed 42 --repeats 1 --device cpu --quick-test
```

Full training:

```bash
hidmoa --dataset neu_xsdd --seed 42 --repeats 1 --device cuda
```

Repeated seeds:

```bash
hidmoa --dataset neu_xsdd --seed 42 --repeats 3 --device cuda
```

## Results

Results are written to:

```text
runs/<timestamp>_HiDMoA/
```

Important files:

```text
test.txt or test_repeat.txt      final accuracy / macro-F1 / per-session metrics
cost.txt or cost_repeat.txt      runtime, parameter, memory, and FLOPs summary
summary.json                     machine-readable metrics
router.txt                       task-router diagnostics
config.json                      full runtime config snapshot
plot/                            loss curves and optional router plots
models/                          saved checkpoints, if enabled by the run path
```

For `--repeats 1`, read `test.txt`. For `--repeats > 1`, read
`test_repeat.txt`, which reports mean and standard deviation over the seed list.

## Important Hyperparameters

Most method hyperparameters are in `INCREMENTAL_2` inside `src/hidmoa/config.py`.
The most commonly changed settings are:

```text
`*` means key parameters.
CIL_ACTIVE_DATASET*                  dataset/task preset name
CIL_SEED*                           base seed
CIL_REPEATS*                        number of repeated seeds
HIDMOA_DEVICE                       cuda or cpu
HIDMOA_BATCH_SIZE*                  model training batch size
CIL_NUM_WORKERS                      DataLoader workers; default 0 for portability
HIDMOA_EPOCHS_PER_TASK*             epochs for each incremental session
HIDMOA_EARLY_STOP_PATIENCE*         early stopping patience for session training
HIDMOA_FVAE_EPOCHS*                 feature-VAE router training epochs
HIDMOA_FVAE_PATIENCE*               feature-VAE early stopping patience
HIDMOA_FVAE_BATCH_SIZE              feature-VAE batch size
HIDMOA_FVAE_GENERATED_PER_CLASS*    generated/replayed feature count per class
HIDMOA_FVAE_FEATURE_BATCH           feature extraction batch size
HIDMOA_ROUTER_IMPORTANCE_SAMPLES*   router scoring sample count
CIL_PROFILE_FLOPS                   1 to profile FLOPs, 0 to skip
train.fvae.lr*                      feature-VAE learning rate
train.fvae.weight_decay*            feature-VAE weight decay
train.fvae.h_dim*                   feature-VAE hidden dimension
train.fvae.z_dim*                   feature-VAE latent dimension
```

Key config fields in `INCREMENTAL_2`:

```text
model.backbone*                      resnet18, resnet50, resnet101, etc.
model.pretrained                     use ImageNet-pretrained backbone weights
model.moe_layers                     backbone stages with MoE adapters
model.experts_per_task*              experts added per task
train.lr*                           classifier/adapter learning rate
train.weight_decay*                 classifier/adapter weight decay
train.head_type*                    cosine_prototype, prototype, linear, cos
train.class_internal_loss            disabled by default; lambda=0.0 and supcon_weight=0.0
train.vae_router_type*              fvae by default
train.vae_router_grouping*          task by default
train.vae_router_score_mode*        recon by default
train.vae_router_backbone*          router feature extractor backbone
train.task_router_inference*        top1 by default
train.task_router_alpha*            used for top2 fusion/search variants
train.fvae.h_dim*                   feature-VAE hidden dimension
train.fvae.z_dim*                   feature-VAE latent dimension
train.fvae.beta_kl*                 KL loss weight
```

`train.class_internal_loss` is kept for ablation compatibility. In the default
HiDMoA setting it is not enabled:

```python
"class_internal_loss": {
    "lambda": 0.0,
    "temperature": 0.2,
    "supcon_weight": 0.0,
    "focal_weight": 1.0,
    "focal_gamma": 2.0,
}
```

Because `lambda=0.0` and `supcon_weight=0.0`, the extra class-internal /
supervised-contrastive loss does not contribute to training by default.
