# Dataset selection.
export CIL_ACTIVE_DATASET=neu_xsdd

# Dataset roots. The repository already includes NEU/XSDD under data/.
# Uncomment these only when using your own dataset copies.
# export CIL_NEU_ROOT=/path/to/neudata_yolo_701515
# export CIL_XSDD_ROOT=/path/to/xsdd_yolo_cls_701515

# Reproducibility.
export CIL_SEED=42
export CIL_REPEATS=1

# Runtime.
export HIDMOA_DEVICE=cuda
export HIDMOA_BATCH_SIZE=32
export CIL_NUM_WORKERS=0

# Training length. For a quick smoke test, use:
#   export HIDMOA_EPOCHS_PER_TASK=1
#   export HIDMOA_FVAE_EPOCHS=1
#   export HIDMOA_FVAE_GENERATED_PER_CLASS=8
export HIDMOA_EPOCHS_PER_TASK=50
export HIDMOA_EARLY_STOP_PATIENCE=5
export HIDMOA_FVAE_EPOCHS=200
export HIDMOA_FVAE_PATIENCE=20
export HIDMOA_FVAE_GENERATED_PER_CLASS=600

# Disable FLOPs profiling for faster smoke tests.
export CIL_PROFILE_FLOPS=0
