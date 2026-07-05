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

export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export SLURM_TIME="${SLURM_TIME:-00:40:00}"
export QWEN35_ATTENTION_BACKEND=auto

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

submit_replicas coreai_mlperf_training-grpo.smoke40m-1step-16p1g-tp4pp2cp2ep32-vllmtp8-unpacked-auto-raydag-t1210 1 \
    grpo.num_prompts_per_step=16 \
    grpo.num_generations_per_prompt=1 \
    grpo.max_num_steps=1 \
    grpo.val_at_start=False \
    grpo.val_at_end=False \
    policy.train_global_batch_size=16 \
    policy.generation_batch_size=16 \
    policy.generation.max_new_tokens=512 \
    policy.megatron_cfg.tensor_model_parallel_size=4 \
    policy.megatron_cfg.pipeline_model_parallel_size=2 \
    policy.megatron_cfg.context_parallel_size=2 \
    policy.megatron_cfg.sequence_parallel=true \
    policy.megatron_cfg.expert_model_parallel_size=32 \
    policy.sequence_packing.enabled=false \
    policy.generation.vllm_cfg.tensor_parallel_size=8 \
    policy.generation.vllm_cfg.expert_parallel_size=8 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.80 \
    env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=2 \
    env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.concurrency=16 \
    env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=2 \
    env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.concurrency=16 \
    checkpointing.enabled=False
