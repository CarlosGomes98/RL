#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Do not leak a host virtual environment into the container's uv environment.
unset VIRTUAL_ENV
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_LAUNCH_ORDER_IMPLICIT="${NCCL_LAUNCH_ORDER_IMPLICIT:-1}"

REPO_ROOT="${REPO_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/arigazzi/RL-main}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/examples/nemo_gym/launch_qwen35_nemo_gym_multinode_training.sh}"

export REPO_LOCATION="${REPO_LOCATION:-${REPO_ROOT}}"
DEFAULT_QWEN35_CONTAINER_IMAGE_PATH="/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/containers/dl+mlperf+optimized+nemo-rl-nightly-qwen35-ea7c4f9-fa4576049.sqsh"
export CONTAINER_IMAGE_PATH="${QWEN35_CONTAINER_IMAGE_PATH:-${CONTAINER_IMAGE_PATH:-${DEFAULT_QWEN35_CONTAINER_IMAGE_PATH}}}"
# The image contains the qualified Qwen source. Overlaying the whole host tree
# can replace untouched nightly modules with incompatible branch-era versions.
export NRL_SOURCE_OVERLAY="${NRL_SOURCE_OVERLAY:-0}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-coreai_mlperf_training}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export SLURM_TIME="${SLURM_TIME:-04:00:00}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:${GPUS_PER_NODE}}"
export SBATCH_SEGMENT="${SBATCH_SEGMENT:-8}"
QWEN35_CLUSTER_SEGMENT_SIZE="${QWEN35_CLUSTER_SEGMENT_SIZE:-${SBATCH_SEGMENT}}"
export TRAIN_NODES="${TRAIN_NODES:-32}"
export GEN_NODES="${GEN_NODES:-32}"
export NODES="${NODES:-64}"
QWEN35_GYM_CONCURRENCY="${QWEN35_GYM_CONCURRENCY:-$((GEN_NODES * GPUS_PER_NODE))}"
export RECIPE="${RECIPE:-qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_benchmark.yaml}"
export HF_CKPT_PATH="${HF_CKPT_PATH:-/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/arigazzi/nemotron3_ultra_550b/hf_home/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/8472618112abcbd45acbcdc58436aff4233c23f7}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/arigazzi/nemotron3_ultra_550b/mcore_ckpt_cache}"
export NEMO_GYM_SWE_TRAIN_DATA_PATH="${NEMO_GYM_SWE_TRAIN_DATA_PATH:-/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/hfilaretov/data/Benchmark-R2E-Gym-Easy-Large/benchmark_r2e_gym_easy_train.jsonl}"
export NEMO_GYM_SWE_VALIDATION_DATA_PATH="${NEMO_GYM_SWE_VALIDATION_DATA_PATH:-/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/hfilaretov/data/Benchmark-R2E-Gym-Easy-Large/benchmark_r2e_gym_easy_val.jsonl}"
export NEMO_GYM_SWE_SIF_DIR="${NEMO_GYM_SWE_SIF_DIR:-/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/hfilaretov/data}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/dev/fuse:/dev/fuse}"
export MLPERF_TARGET_ACCURACY="${MLPERF_TARGET_ACCURACY:-1.0}"
export NEMO_RL_MEGATRON_POLICY_CUDA_MEMORY_FRACTION="${NEMO_RL_MEGATRON_POLICY_CUDA_MEMORY_FRACTION:-0.67}"
# Use RayDistributedExecutor's compiled-graph backend. RayExecutorV2's MQ path
# has timed out in sample_tokens during long multi-node Qwen 3.5 rollouts.
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND:-0}"
# Allow long rollout-driven model steps to complete before Ray times out.
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-1210}"
QWEN35_ATTENTION_BACKEND="${QWEN35_ATTENTION_BACKEND:-flash}"
QWEN35_POLICY_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
QWEN35_POLICY_LD_LIBRARY_PATH="${QWEN35_POLICY_VENV}/lib/python3.13/site-packages/torch/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

BASELINE_OVERRIDES=(
    logger.wandb_enabled=False
    ++cluster.segment_size="${QWEN35_CLUSTER_SEGMENT_SIZE}"
    grpo.async_grpo.max_trajectory_age_steps=1
    env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.concurrency="${QWEN35_GYM_CONCURRENCY}"
    env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.concurrency="${QWEN35_GYM_CONCURRENCY}"
    # vLLM 0.20 can deadlock Qwen 3.5 multi-node decode with CUDA graphs.
    policy.generation.vllm_cfg.enforce_eager=True
    ++policy.megatron_cfg.attention_backend="${QWEN35_ATTENTION_BACKEND}"
    # DeepEP is loaded only by Megatron policy actors. Keep their pinned Torch
    # libraries visible without changing library resolution for vLLM or Gym.
    ++policy.megatron_cfg.env_vars.LD_LIBRARY_PATH="${QWEN35_POLICY_LD_LIBRARY_PATH}"
    policy.megatron_cfg.expert_model_parallel_size=32
    policy.megatron_cfg.scheduler.lr_warmup_iters=0
)

make_seed() {
    od -An -N4 -tu4 /dev/urandom | tr -d " "
}

next_replica() {
    local group="$1"
    local group_dir="${RESULTS_ROOT}/${group}"
    local next=1
    local path base n

    for path in "${group_dir}"/r* "${RESULTS_ROOT}/${group}"-r*; do
        [[ -e "${path}" ]] || continue
        base="$(basename "${path}")"
        if [[ "${base}" =~ ^r([0-9]+)$ ]]; then
            n="${BASH_REMATCH[1]}"
        elif [[ "${base}" =~ -r([0-9]+)$ ]]; then
            n="${BASH_REMATCH[1]}"
        else
            continue
        fi
        if ((n >= next)); then
            next=$((n + 1))
        fi
    done

    printf "%s\n" "${next}"
}

submit_replica() {
    local group="$1"
    local replica="$2"
    shift 2

    local run_name="${group}-r${replica}"
    local group_dir="${RESULTS_ROOT}/${group}"
    local seed="${GRPO_SEED_OVERRIDE:-$(make_seed)}"
    local output job_id

    mkdir -p "${group_dir}"
    if [[ ! -f "${group_dir}/manifest.tsv" ]]; then
        printf "submitted_at\tgroup\treplica\texp_name\tjob_id\tseed\toverrides\n" > "${group_dir}/manifest.tsv"
    fi

    echo "Submitting ${run_name} with grpo.seed=${seed}"
    if ! output=$(EXP_NAME="${run_name}" SLURM_JOB_NAME="${run_name}" GRPO_SEED="${seed}" "${LAUNCHER}" "${BASELINE_OVERRIDES[@]}" "$@" 2>&1); then
        printf "%s\n" "${output}"
        return 1
    fi
    printf "%s\n" "${output}"

    job_id="$(printf "%s\n" "${output}" | awk "/Submitted batch job/ {print \$4}" | tail -n 1)"
    ln -sfn "../${run_name}" "${group_dir}/r${replica}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date -Is)" "${group}" "r${replica}" "${run_name}" "${job_id:-unknown}" "${seed}" "$*" \
        >> "${group_dir}/manifest.tsv"
}

submit_replicas() {
    local group="$1"
    local count="$2"
    shift 2

    local first i
    first="$(next_replica "${group}")"
    for ((i = 0; i < count; i++)); do
        submit_replica "${group}" "$((first + i))" "$@"
    done

    echo "Group directory: ${RESULTS_ROOT}/${group}"
}
