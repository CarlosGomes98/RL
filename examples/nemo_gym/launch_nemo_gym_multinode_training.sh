# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# ----- PARAMETERS -----
# WANDB_API_KEY, HF_TOKEN, EXP_NAME, RECIPE, TRAIN_NODES, GEN_NODES, REPO_LOCATION, CONTAINER_IMAGE_PATH, SLURM_ACCOUNT, SLURM_PARTITION

TRAIN_NODES="${TRAIN_NODES:-16}"
GEN_NODES="${GEN_NODES:-24}"
NODES="${NODES:-$((TRAIN_NODES + GEN_NODES))}"
CONTAINER_REPO_LOCATION="${CONTAINER_REPO_LOCATION:-/opt/nemo-rl}"
RECIPE="${RECIPE:-examples/nemo_gym/grpo_qwen3_235b_swe_openhands_async.yaml}"

# ray.sub is submitted from the host checkout, but training runs from the
# baked checkout inside the container.
cd $REPO_LOCATION
OUT_DIR="$(pwd)/results/${EXP_NAME}"
HOST_HF_HOME="${HF_HOME:-$(pwd)/.cache}"
export BASE_LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/checkpoint" "${HOST_HF_HOME}"

# Construct the command
COMMAND=$(cat <<EOF
cd ${CONTAINER_REPO_LOCATION}


HF_HOME=${CONTAINER_REPO_LOCATION}/.cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_TOKEN=$HF_TOKEN \
WANDB_API_KEY=$WANDB_API_KEY \
NRL_MEGATRON_CHECKPOINT_DIR=$NRL_MEGATRON_CHECKPOINT_DIR \
NEMO_GYM_SWE_WORKSPACE_ROOT=/logs/nemo_gym/workspace \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config ${RECIPE} \
    ++cluster.num_nodes=$NODES \
    ++policy.generation.colocated.resources.num_nodes=$GEN_NODES \
    ++logger.wandb.name=$EXP_NAME \
    ++logger.log_dir=/logs \
    ++checkpointing.checkpoint_dir=/checkpoint \
    $@
EOF
)

echo -e "Running command:\n$COMMAND"

mount=$(findmnt -n -o TARGET --target .)


# OccupiedIdleGPUsJobReaper exemption: async (non-colocated) legitimately idles its
# training-node GPU pool while the replay buffer fills from slow SWE reward computation,
# which otherwise trips the idle-GPU reaper. Override via SLURM_COMMENT env if needed.
SLURM_IDLE_EXEMPT_MINS="${SLURM_IDLE_EXEMPT_MINS:-120}"
SLURM_COMMENT="${SLURM_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"${SLURM_IDLE_EXEMPT_MINS}\",\"reason\":\"rl-rollout-warmup\",\"description\":\"NeMo-RL GRPO: training GPUs idle during rollout/SWE-reward buffer-fill\"}}}"

COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
CONTAINER_WORKDIR=$CONTAINER_REPO_LOCATION \
MOUNTS=$mount:$mount,${HOST_HF_HOME}:${CONTAINER_REPO_LOCATION}/.cache,${OUT_DIR}/checkpoint:/checkpoint,${OUT_DIR}/logs:/logs \
sbatch \
    --nodes=$NODES \
    --account=$SLURM_ACCOUNT \
    --partition=$SLURM_PARTITION \
    --time=${SLURM_TIME:-1:0:0} \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    --comment="$SLURM_COMMENT" \
    ${SLURM_EXCLUDE:+--exclude=$SLURM_EXCLUDE} \
    ray.sub
