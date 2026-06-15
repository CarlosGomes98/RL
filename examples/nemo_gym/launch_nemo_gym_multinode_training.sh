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

# ----- PARAMETERS -----
# WANDB_API_KEY, HF_TOKEN, EXP_NAME, NUM_ACTOR_NODES, NUM_SLURM_NODES (optional), REPO_LOCATION, CONTAINER_IMAGE_PATH, SLURM_ACCOUNT, SLURM_PARTITION

# ray.sub needs to be launched from the NeMo-RL root directory
cd $REPO_LOCATION

# Construct the command
read -r -d '' COMMAND <<EOF
cd ${REPO_LOCATION}

HF_HOME=$PWD/.cache/ \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_TOKEN=$HF_TOKEN \
WANDB_API_KEY=$WANDB_API_KEY \
NRL_MEGATRON_CHECKPOINT_DIR=$NRL_MEGATRON_CHECKPOINT_DIR \
NEMO_GYM_SWE_WORKSPACE_ROOT=$NEMO_GYM_SWE_WORKSPACE_ROOT \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    ++cluster.num_nodes=$NUM_ACTOR_NODES \
    ++logger.wandb.name=$EXP_NAME \
    ++logger.log_dir=results/$EXP_NAME \
    ++checkpointing.checkpoint_dir=results/$EXP_NAME \
    $@
EOF

echo -e "Running command:\n$COMMAND"

mount=$(findmnt -n -o TARGET --target .)

FINAL_NUM_SLURM_NODES="${NUM_SLURM_NODES:-$NUM_ACTOR_NODES}"

# OccupiedIdleGPUsJobReaper exemption: async (non-colocated) legitimately idles its
# training-node GPU pool while the replay buffer fills from slow SWE reward computation,
# which otherwise trips the idle-GPU reaper. Override via SLURM_COMMENT env if needed.
SLURM_IDLE_EXEMPT_MINS="${SLURM_IDLE_EXEMPT_MINS:-120}"
SLURM_COMMENT="${SLURM_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"${SLURM_IDLE_EXEMPT_MINS}\",\"reason\":\"rl-rollout-warmup\",\"description\":\"NeMo-RL GRPO: training GPUs idle during rollout/SWE-reward buffer-fill\"}}}"

COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
MOUNTS=$mount:$mount \
sbatch \
    --nodes=$FINAL_NUM_SLURM_NODES \
    --account=$SLURM_ACCOUNT \
    --partition=$SLURM_PARTITION \
    --time=${SLURM_TIME:-1:0:0} \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    --comment="$SLURM_COMMENT" \
    ${SLURM_EXCLUDE:+--exclude=$SLURM_EXCLUDE} \
    ray.sub
