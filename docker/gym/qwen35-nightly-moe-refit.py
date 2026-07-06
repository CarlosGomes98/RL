# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from typing import Any

import torch


def _weight_summary(name: str, weight: torch.Tensor) -> str:
    return (
        f"name={name!r}, shape={tuple(weight.shape)}, dtype={weight.dtype}, "
        f"device={weight.device}, stride={weight.stride()}"
    )


def _qwen35_moe_export_spec(name: str) -> tuple[int, str] | None:
    for prefix in (
        "model.language_model.model.",
        "model.language_model.",
        "language_model.model.",
        "language_model.",
    ):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break

    parts = name.split(".")
    if (
        len(parts) not in (5, 6)
        or parts[0] != "layers"
        or parts[2:4] != ["mlp", "experts"]
    ):
        return None
    if len(parts) == 6 and parts[5] != "weight":
        return None
    if parts[4] not in {"gate_up_proj", "down_proj"}:
        return None

    try:
        return int(parts[1]), parts[4]
    except ValueError:
        return None


def _split_qwen35_moe_weights(
    policy_weights: list[tuple[str, torch.Tensor]], model_runner: Any
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor, int, str]]]:
    if not getattr(model_runner.model, "is_3d_moe_weight", False):
        return policy_weights, []

    normal_weights = []
    moe_weights = []
    for name, weight in policy_weights:
        spec = _qwen35_moe_export_spec(name)
        if spec is not None and weight.ndim == 3:
            layer_idx, kind = spec
            moe_weights.append((name, weight, layer_idx, kind))
        else:
            normal_weights.append((name, weight))
    return normal_weights, moe_weights


def _get_inner_model(model_runner: Any) -> Any:
    language_model = getattr(model_runner.model, "language_model", None)
    inner_model = getattr(language_model, "model", None)
    if inner_model is None:
        inner_model = getattr(model_runner.model, "model", None)
    return inner_model


def _get_fused_moe_param(experts: Any, name: str) -> torch.nn.Parameter | None:
    for owner in (experts, getattr(experts, "base_layer", None)):
        if owner is not None and (param := getattr(owner, name, None)) is not None:
            return param
    return None


def _copy_expert_tensor(
    param: torch.nn.Parameter,
    local_expert_id: int,
    weight: torch.Tensor,
    name: str,
) -> None:
    if not 0 <= local_expert_id < param.data.shape[0]:
        raise RuntimeError(
            f"Qwen3.5 local expert {local_expert_id} is outside destination "
            f"shape {tuple(param.data.shape)} for {_weight_summary(name, weight)}"
        )

    destination = param.data[local_expert_id]
    if destination.ndim == 3 and weight.ndim == 2:
        if (
            destination.shape[1] == weight.shape[0]
            and destination.shape[0] * destination.shape[2] == weight.shape[1]
        ):
            weight = (
                weight.view(weight.shape[0], destination.shape[0], destination.shape[2])
                .permute(1, 0, 2)
                .contiguous()
            )
        elif (
            destination.shape[1] == weight.shape[1]
            and destination.shape[0] * destination.shape[2] == weight.shape[0]
        ):
            weight = (
                weight.view(
                    destination.shape[0], destination.shape[2], weight.shape[1]
                )
                .permute(0, 2, 1)
                .contiguous()
            )

    if destination.ndim != weight.ndim:
        raise RuntimeError(
            "Qwen3.5 expert refit tensor rank mismatch: "
            f"destination={tuple(destination.shape)}, {_weight_summary(name, weight)}"
        )

    slices = []
    for dim, (destination_size, weight_size) in enumerate(
        zip(destination.shape, weight.shape)
    ):
        if weight_size > destination_size:
            raise RuntimeError(
                "Qwen3.5 expert refit tensor shape mismatch: "
                f"destination={tuple(destination.shape)}, "
                f"{_weight_summary(name, weight)}, dim={dim}"
            )
        slices.append(slice(0, weight_size))

    destination = destination[tuple(slices)]
    destination.copy_(weight.to(device=destination.device, dtype=destination.dtype))


def _load_qwen35_moe_weights(
    moe_weights: list[tuple[str, torch.Tensor, int, str]], model_runner: Any
) -> None:
    if not moe_weights:
        return

    inner_model = _get_inner_model(model_runner)
    layers = getattr(inner_model, "layers", None)
    if layers is None:
        name, weight, _, _ = moe_weights[0]
        raise RuntimeError(
            "Qwen3.5 refit could not find vLLM language model layers for "
            f"{_weight_summary(name, weight)}"
        )

    start_layer = getattr(inner_model, "start_layer", 0)
    end_layer = getattr(inner_model, "end_layer", len(layers))

    copied = 0
    with torch.no_grad():
        for name, weight, layer_idx, kind in moe_weights:
            if not start_layer <= layer_idx < end_layer:
                continue

            try:
                experts = layers[layer_idx].mlp.experts
            except (AttributeError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"Qwen3.5 refit could not find fused experts for layer {layer_idx}; "
                    f"{_weight_summary(name, weight)}"
                ) from exc

            param_name = "w13_weight" if kind == "gate_up_proj" else "w2_weight"
            param = _get_fused_moe_param(experts, param_name)
            if param is None:
                raise RuntimeError(
                    f"Qwen3.5 refit could not find experts.{param_name} for layer "
                    f"{layer_idx}; {_weight_summary(name, weight)}"
                )

            mapper = getattr(experts, "_map_global_expert_id_to_local_expert_id", None)
            for global_expert_id in range(weight.shape[0]):
                local_expert_id = (
                    mapper(global_expert_id) if mapper is not None else global_expert_id
                )
                if local_expert_id == -1:
                    continue
                _copy_expert_tensor(
                    param, local_expert_id, weight[global_expert_id], name
                )
                copied += 1

    if copied == 0:
        name, weight, _, _ = moe_weights[0]
        raise RuntimeError(
            "Qwen3.5 refit found fused expert exports but copied no local experts; "
            f"{_weight_summary(name, weight)}"
        )

    name, weight, _, _ = moe_weights[0]
    print(
        f"[qwen35_moe_refit] copied {copied} local experts from "
        f"{len(moe_weights)} tensors; first={_weight_summary(name, weight)}",
        flush=True,
    )


def load_policy_weights(
    policy_weights: list[tuple[str, torch.Tensor]], model_runner: Any
) -> None:
    """Load normal weights through vLLM and Qwen3.5 fused experts directly."""
    normal_weights, moe_weights = _split_qwen35_moe_weights(
        policy_weights, model_runner
    )

    if normal_weights:
        try:
            model_runner.model.load_weights(weights=normal_weights)
        except Exception as original_error:
            for name, weight in normal_weights:
                try:
                    model_runner.model.load_weights(weights=[(name, weight)])
                except Exception as single_error:
                    raise RuntimeError(
                        "vLLM load_weights failed for "
                        f"{_weight_summary(name, weight)}; "
                        f"single_error={single_error!r}; "
                        f"batch_error={original_error!r}"
                    ) from original_error
            raise

    _load_qwen35_moe_weights(moe_weights, model_runner)
