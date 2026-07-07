#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export SLURM_TIME="${SLURM_TIME:-01:00:00}"
export TRAIN_NODES=16
export GEN_NODES=48
export NODES=64
export QWEN35_ATTENTION_BACKEND=flash
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

submit_replicas coreai_mlperf_training-grpo.smoke1h-gbs512-16p48g-lr1p25e-6-1step-tp4pp2cp1ep32-hybridep-vllmtp8pp1ep8-raydag-cgraph-packed-flash-noprefixcache-agentt1200-native-loader 1 \
    grpo.num_prompts_per_step=32 \
    grpo.num_generations_per_prompt=16 \
    policy.train_global_batch_size=512 \
    policy.megatron_cfg.tensor_model_parallel_size=4 \
    policy.megatron_cfg.pipeline_model_parallel_size=2 \
    policy.megatron_cfg.expert_model_parallel_size=32 \
    policy.megatron_cfg.context_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=true \
    policy.sequence_packing.enabled=true \
    policy.generation.vllm_cfg.tensor_parallel_size=8 \
    policy.generation.vllm_cfg.pipeline_parallel_size=1 \
    policy.generation.vllm_cfg.expert_parallel_size=8 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.80 \
    policy.generation.vllm_cfg.enforce_eager=False \
    ++policy.generation.vllm_kwargs.compilation_config.mode=0 \
    ++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_DECODE_ONLY \
    ++policy.generation.vllm_cfg.enable_prefix_caching=false \
    env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=1200 \
    env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=1200 \
    grpo.max_num_steps=1 \
    grpo.val_period=2 \
    grpo.val_at_start=False \
    grpo.val_at_end=False \
    policy.megatron_cfg.optimizer.lr=1.25e-6 \
    policy.megatron_cfg.optimizer.min_lr=1.25e-6 \
    checkpointing.enabled=False
