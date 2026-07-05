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

export SLURM_PARTITION="${SLURM_PARTITION:-batch_long}"
export SLURM_TIME="${SLURM_TIME:-06:00:00}"

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

submit_replicas coreai_mlperf_training-grpo.gbs512-32p16g-lr1p5e-6-valp2-steps10-ckpt7-tp4pp2cp1ep32-flash 1 \
    grpo.num_prompts_per_step=32 \
    grpo.num_generations_per_prompt=16 \
    policy.train_global_batch_size=512 \
    policy.megatron_cfg.tensor_model_parallel_size=4 \
    policy.megatron_cfg.pipeline_model_parallel_size=2 \
    policy.megatron_cfg.expert_model_parallel_size=32 \
    policy.megatron_cfg.context_parallel_size=1 \
    grpo.max_num_steps=10 \
    grpo.val_period=2 \
    grpo.val_at_start=False \
    grpo.val_at_end=False \
    policy.megatron_cfg.optimizer.lr=1.5e-6 \
    policy.megatron_cfg.optimizer.min_lr=1.5e-6 \
    checkpointing.enabled=True \
    checkpointing.save_period=7
