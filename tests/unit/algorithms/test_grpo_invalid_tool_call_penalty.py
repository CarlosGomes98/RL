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

from unittest.mock import patch

import pytest
import torch

from nemo_rl.algorithms.grpo import (
    _apply_configured_invalid_tool_call_advantage_penalty,
    _apply_invalid_tool_call_advantage_penalty,
    _resolve_invalid_tool_call_advantage,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_apply_invalid_tool_call_advantage_penalty_targets_flagged_message_spans():
    expanded_advantages = torch.tensor([[0.0], [10.0]]).expand(2, 6)
    train_data = BatchedDataDict({"advantages": expanded_advantages})
    message_logs = [
        [
            {
                "role": "user",
                "token_ids": torch.tensor([1, 2]),
                "is_invalid_tool_call": True,
            },
            {
                "role": "assistant",
                "token_ids": torch.tensor([3, 4]),
                "generation_logprobs": torch.tensor([0.1, 0.2]),
                "is_invalid_tool_call": True,
            },
            {
                "role": "assistant",
                "token_ids": torch.tensor([5, 6]),
                "generation_logprobs": torch.tensor([0.3, 0.4]),
            },
        ],
        [
            {
                "role": "user",
                "token_ids": torch.tensor([7]),
            },
            {
                "role": "assistant",
                "token_ids": torch.tensor([8, 9, 10]),
                "generation_logprobs": torch.tensor([0.5, 0.6, 0.7]),
                "is_invalid_tool_call": True,
            },
            {
                "role": "assistant",
                "token_ids": torch.tensor([11, 12]),
                "generation_logprobs": torch.tensor([0.8, 0.9]),
            },
        ],
    ]

    _apply_invalid_tool_call_advantage_penalty(
        train_data=train_data,
        message_logs=message_logs,
        invalid_tool_call_advantage=-5.0,
    )

    expected = torch.tensor(
        [
            [0.0, 0.0, -5.0, -5.0, 0.0, 0.0],
            [10.0, -5.0, -5.0, -5.0, 10.0, 10.0],
        ]
    )
    torch.testing.assert_close(train_data["advantages"], expected)


def test_apply_configured_invalid_tool_call_advantage_penalty_noops_when_disabled():
    train_data = BatchedDataDict(
        {"advantages": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}
    )
    message_logs = [
        [
            {
                "role": "assistant",
                "token_ids": torch.tensor([1, 2]),
                "generation_logprobs": torch.tensor([0.1, 0.2]),
                "is_invalid_tool_call": True,
            }
        ]
    ]
    master_config = {"grpo": {}}

    with patch("nemo_rl.algorithms.grpo._should_use_nemo_gym") as should_use_nemo_gym:
        _apply_configured_invalid_tool_call_advantage_penalty(
            train_data, message_logs, master_config
        )

    should_use_nemo_gym.assert_not_called()
    torch.testing.assert_close(
        train_data["advantages"], torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    )


def test_resolve_invalid_tool_call_advantage_requires_nemo_gym():
    master_config = {"grpo": {"invalid_tool_call_advantage": -5.0}}

    with patch("nemo_rl.algorithms.grpo._should_use_nemo_gym", return_value=False):
        with pytest.raises(AssertionError, match="NeMo-Gym path"):
            _resolve_invalid_tool_call_advantage(master_config)
