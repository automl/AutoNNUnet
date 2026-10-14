#!/usr/bin/env bash
# Launches runscripts/train.py inside the autonnunet:local Docker image for
# local (non-SLURM) runs - single training jobs, smoke tests, or
# `cluster=local_multi_gpu` sweeps.
#
# This exists so the flags that matter for correctness/hygiene on a long
# unattended run don't depend on typing them correctly by hand each time:
#   --init       Docker's built-in tini becomes PID 1 and reaps orphaned
#                child processes. Without it, submitit's per-job controller
#                processes accumulate as zombies for the life of the
#                container (harmless in small numbers, but unbounded on a
#                multi-day run).
#   --shm-size   Large enough for concurrent dataloader worker IPC across
#                several GPUs at once; default sized for up to ~6 concurrent
#                jobs, override for more.
#
# Usage:
#   runscripts/run_docker.sh <container_name> <gpu_csv> [shm_size] -- <args to train.py>
#
# Example (6-GPU RAM-cached sweep):
#   runscripts/run_docker.sh autonnunet_hpo_nas 0,1,2,3,4,5 96g -- \
#     --config-name=tune_hpo_nas -m dataset=Dataset003_CT_Teeth_corrected_postprocessed \
#     cluster=local_multi_gpu trainer.cache_dataset_in_ram=true
#
# Example (single-GPU smoke test):
#   runscripts/run_docker.sh autonnunet_smoke_test 0 32g -- \
#     --config-name=train dataset=Dataset003_CT_Teeth_corrected_postprocessed cluster=local

set -euo pipefail

if [ "$#" -lt 4 ] || [ "$3" == "--" ]; then
    # shm_size omitted: shift args, use default
    CONTAINER_NAME="$1"
    GPUS="$2"
    SHM_SIZE="64g"
    shift 2
else
    CONTAINER_NAME="$1"
    GPUS="$2"
    SHM_SIZE="$3"
    shift 3
fi

if [ "${1:-}" != "--" ]; then
    echo "Usage: $0 <container_name> <gpu_csv> [shm_size] -- <args to train.py>" >&2
    exit 1
fi
shift  # drop the --

# These paths are specific to this host; adjust if running elsewhere.
NNUNET_RAW_HOST="/home/dombeckt/hpc_ag-ml/rescue-defacing-models/nnUNet/nnUNet_raw"
NNUNET_PREPROCESSED_HOST="/home/dombeckt/hpc_my-share/autonnunet_preprocessed"
NNUNET_RESULTS_HOST="/home/dombeckt/hpc_my-share/autonnunet_results"
REPO_HOST="/home/dombeckt/projects/AutoNNUnet/repo"
OUTPUT_HOST="/home/dombeckt/ml_workgroup/project/rescue-defacing-models/AutoNNUnet/output"

docker run -d \
    --name "$CONTAINER_NAME" \
    --init \
    --gpus "\"device=${GPUS}\"" \
    --shm-size="$SHM_SIZE" \
    -v "${NNUNET_RAW_HOST}:/tmp/autonnunet/data/nnUNet_raw" \
    -v "${NNUNET_PREPROCESSED_HOST}:/tmp/autonnunet/data/nnUNet_preprocessed" \
    -v "${NNUNET_RESULTS_HOST}:/tmp/autonnunet/data/nnUNet_results" \
    -v "${REPO_HOST}:/tmp/autonnunet" \
    -v "${OUTPUT_HOST}:/tmp/autonnunet/output" \
    autonnunet:local \
    bash -c "cd /tmp/autonnunet && python3 runscripts/train.py $*"

echo "Started container: ${CONTAINER_NAME}"
