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

import threading as _threading
import time
from collections import Counter
from typing import Any, Iterable, Optional

import ray
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface

TokenizerType = PreTrainedTokenizerBase


class ReplayBufferImpl:
    """Replay buffer storing per-prompt groups.

    A single entry corresponds to 1 prompt repeated by
    grpo.num_generations_per_prompt (required to compute per-prompt advantages).
    """

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self.trajectories = []  # List[dict[str, Any]]
        # If trajectory_version is 1 and target_weight_version is 4 it means that weight version 1 was used for generating a trajectory and this trajectory will be used for training when weight version is 4.
        self.trajectory_versions = []  # it is the weight-version used for generation of a trajectory
        self.target_weight_versions = []  # it is the weight-version of the trainer where this trajectory will be used.

        self.last_target_weight_already_generated = -1
        self._lock = _threading.Lock()

    def add(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: int,
    ) -> str:
        """Add a per-prompt trajectory group with metadata.

        Args:
            trajectory: data dict
            weight_version: version of the model weights used for generation
            target_weight_version: version of the model weights this trajectory is intended for training
        """
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            print("🔍 ReplayBuffer.add: Adding trajectory")
            self.trajectories.append(trajectory)
            self.trajectory_versions.append(weight_version)
            self.target_weight_versions.append(target_weight_version)
            # A target is safe to skip only after training consumes a full batch.
            print(
                f"ReplayBuffer state: {len(self.trajectories)} groups, versions={self.trajectory_versions}, targets={self.target_weight_versions}, last_target_weight_already_generated={self.last_target_weight_already_generated}"
            )
            return "success"

    def push_with_wait_signal(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: int,
    ) -> str:
        """Compatibility alias for older async GRPO callers."""
        return self.add(trajectory, weight_version, target_weight_version)

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        return {
            "total_trajectories": len(self.trajectories),
            "trajectory_versions": self.trajectory_versions,
            "target_weight_versions": self.target_weight_versions,
            "max_size": self.max_size,
        }

    def get_last_target_weight_already_generated(self) -> int:
        with self._lock:
            return self.last_target_weight_already_generated

    def get_existing_target_weights(self) -> set[int]:
        """Get set of target weight versions that already have trajectories."""
        with self._lock:
            return set(self.target_weight_versions)

    def _remove_indices(self, indices: Iterable[int]) -> None:
        """Remove trajectories at the given indices."""
        for idx in sorted(indices, reverse=True):
            self.trajectory_versions.pop(idx)
            self.target_weight_versions.pop(idx)
            self.trajectories.pop(idx)

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups intended for the current training step.

        Only returns trajectories with target_weight_version == current_weight_version.
        If insufficient trajectories are available, returns None to stall training
        until the remaining trajectories are generated. This ensures no trajectory
        loses its last chance to be used for its intended training step.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None if insufficient data
        """
        with self._lock:
            if not self.trajectories:
                return None

            total_trajectories = len(self.trajectories)
            print("🔍 ReplayBuffer sampling debug:")
            print(f"   {current_weight_version=}, {max_age_steps=}")
            print(f"   {self.trajectory_versions=}")

            version_counts = Counter(self.trajectory_versions)
            print(f"   {version_counts=}")

            # Compute minimum valid version based on age window
            # max_age_steps=1 means trajectories from the last 1 step are valid
            min_valid_version = max(0, current_weight_version - max_age_steps)
            print(f"   {min_valid_version=}")

            # Evict old trajectories that are beyond the age window. This can
            # happen after checkpoint restore when old trajectories remain.
            old_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if v < min_valid_version
            ]
            if old_indices:
                print(
                    f"   Evicting {len(old_indices)} stale trajectories "
                    f"(version < {min_valid_version})"
                )
                self._remove_indices(old_indices)
                total_trajectories = len(self.trajectories)

            # Filter for valid trajectories without modifying the buffer
            valid_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if min_valid_version <= v <= current_weight_version
            ]
            print(
                f"   valid_indices: {len(valid_indices)}/{total_trajectories} trajectories within age window"
            )
            if not valid_indices:
                print("No trajectories available for sampling.")
                return None

            # Enforce exact number of groups if available; otherwise, signal to wait
            if len(valid_indices) < num_prompt_groups:
                print(
                    f"Insufficient valid groups: have {len(valid_indices)}, need {num_prompt_groups}. Waiting for buffer to fill."
                )
                return None

            # Only select trajectories intended for the current training step
            # This ensures no trajectory loses its "last chance" to be used for its intended step
            intended_indices = [
                i
                for i in valid_indices
                if self.target_weight_versions[i] == current_weight_version
            ]

            print(
                f"   🎯 Found {len(intended_indices)} trajectories intended for current step {current_weight_version}"
            )

            # Stall training if we don't have enough trajectories intended for this step
            if len(intended_indices) < num_prompt_groups:
                print(
                    f"   ⏸️ STALLING: Need {num_prompt_groups} trajectories for step {current_weight_version}, but only {len(intended_indices)} are ready"
                )
                print(
                    f"   ⏸️ Training will wait for remaining {num_prompt_groups - len(intended_indices)} trajectories to be generated"
                )
                return None

            # Select exactly the trajectories intended for this step (FIFO within same target)
            selected: list[int] = intended_indices[:num_prompt_groups]
            print(
                f"   ✅ Selected {len(selected)} trajectories all intended for step {current_weight_version}"
            )

            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )
            print(
                f"✅ Selected counts by generation weight-version: {Counter(sampled_weights)}"
            )
            print(f"📊 Average trajectory age: {avg_trajectory_age:.2f} steps")
            print(
                f"🎯 All selected trajectories target step {current_weight_version} (100% target match)"
            )

            sampled_items = [self.trajectories[i] for i in selected]
            self._remove_indices(selected)

            old_last_target = self.last_target_weight_already_generated
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated,
                current_weight_version,
            )
            if self.last_target_weight_already_generated > old_last_target:
                print(
                    "Advanced last_target_weight_already_generated: "
                    f"{old_last_target} -> "
                    f"{self.last_target_weight_already_generated} "
                    f"(consumed batch for step {current_weight_version})"
                )
            print(
                f"🗑️ Consumed and removed {len(selected)} groups from buffer, old buffer size: {total_trajectories}, new buffer size: {len(self.trajectories)}, new target weight versions {self.target_weight_versions}"
            )

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
            }

    def size(self) -> int:
        """Return current buffer size."""
        with self._lock:
            return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.trajectories.clear()
            self.trajectory_versions.clear()
            self.target_weight_versions.clear()

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""
        with self._lock:
            return {
                "trajectories": list(self.trajectories),
                "trajectory_versions": list(self.trajectory_versions),
                "target_weight_versions": list(self.target_weight_versions),
                "last_target_weight_already_generated": (
                    self.last_target_weight_already_generated
                ),
                "max_size": self.max_size,
            }

    def load_state_dict(
        self,
        state: dict[str, Any],
        num_prompts_per_step: int | None = None,
        current_training_step: int | None = None,
        max_age_steps: int | None = None,
    ) -> None:
        """Restore replay buffer state from a checkpoint."""
        with self._lock:
            required_keys = {
                "trajectories",
                "trajectory_versions",
                "target_weight_versions",
                "last_target_weight_already_generated",
            }
            missing_keys = required_keys - set(state)
            if missing_keys:
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

            trajectories = list(state["trajectories"])
            trajectory_versions = list(state["trajectory_versions"])
            target_weight_versions = list(state["target_weight_versions"])
            if not (
                len(trajectories)
                == len(trajectory_versions)
                == len(target_weight_versions)
            ):
                raise ValueError(
                    "Checkpoint has inconsistent replay buffer lengths: "
                    f"trajectories={len(trajectories)}, "
                    f"trajectory_versions={len(trajectory_versions)}, "
                    f"target_weight_versions={len(target_weight_versions)}"
                )

            if "max_size" in state and state["max_size"] != self.max_size:
                print(
                    "ReplayBuffer max_size changed: "
                    f"checkpoint={state['max_size']}, current={self.max_size}. "
                    "Using current config value."
                )

            self.trajectories = trajectories
            self.trajectory_versions = trajectory_versions
            self.target_weight_versions = target_weight_versions
            self.last_target_weight_already_generated = state[
                "last_target_weight_already_generated"
            ]

            if current_training_step is not None and num_prompts_per_step is not None:
                self._prepare_for_training_step(
                    current_step=current_training_step,
                    num_prompts_per_step=num_prompts_per_step,
                )
            elif num_prompts_per_step is not None and self.trajectories:
                self._remove_incomplete_target_steps(num_prompts_per_step)

            if max_age_steps is not None and self.trajectories:
                self._remove_stale_trajectories(max_age_steps)
                if current_training_step is None and num_prompts_per_step is not None:
                    self._remove_incomplete_target_steps(num_prompts_per_step)

            self._truncate_to_max_size(current_training_step)

            print(
                f"ReplayBuffer restored: {len(self.trajectories)} trajectories, "
                "last_target_weight_already_generated="
                f"{self.last_target_weight_already_generated}"
            )

    def _prepare_for_training_step(
        self, current_step: int, num_prompts_per_step: int
    ) -> None:
        """Prepare restored state so training can resume at ``current_step``."""
        print(f"   Preparing replay buffer for training step {current_step}...")

        original_count = len(self.trajectories)
        indices_to_keep = [
            i
            for i, target in enumerate(self.target_weight_versions)
            if target >= current_step
        ]

        if len(indices_to_keep) < original_count:
            removed_past = original_count - len(indices_to_keep)
            self.trajectories = [self.trajectories[i] for i in indices_to_keep]
            self.trajectory_versions = [
                self.trajectory_versions[i] for i in indices_to_keep
            ]
            self.target_weight_versions = [
                self.target_weight_versions[i] for i in indices_to_keep
            ]
            print(
                f"   Removed {removed_past} trajectories for past steps "
                f"(target < {current_step})"
            )

        if not self.trajectories:
            self.last_target_weight_already_generated = current_step - 1
            print(
                "   No restored trajectories remain; collector will generate "
                f"from step {current_step}"
            )
            return

        target_counts = Counter(self.target_weight_versions)
        complete_targets = {
            target
            for target, count in target_counts.items()
            if count >= num_prompts_per_step
        }
        incomplete_targets = {
            target
            for target, count in target_counts.items()
            if count < num_prompts_per_step
        }

        print(
            "   Complete targets: "
            f"{sorted(complete_targets) if complete_targets else 'none'}"
        )
        for target in sorted(incomplete_targets):
            print(
                f"   Incomplete target {target}: "
                f"{target_counts[target]}/{num_prompts_per_step}"
            )

        # Let the collector gap-fill incomplete restored batches and skip
        # targets that already have complete batches.
        self.last_target_weight_already_generated = current_step - 1

    @staticmethod
    def _is_valid_for_target(
        trajectory_version: int, target_step: int, max_age_steps: int | None
    ) -> bool:
        if max_age_steps is None:
            return True
        min_valid_version = max(0, target_step - max_age_steps)
        return min_valid_version <= trajectory_version <= target_step

    def _remove_stale_trajectories(self, max_age_steps: int) -> None:
        """Remove restored trajectories that are stale for their target step."""
        indices_to_remove = [
            i
            for i, (trajectory_version, target) in enumerate(
                zip(self.trajectory_versions, self.target_weight_versions)
            )
            if not self._is_valid_for_target(trajectory_version, target, max_age_steps)
        ]
        if not indices_to_remove:
            return

        print(
            f"   Removing {len(indices_to_remove)} stale restored trajectories "
            f"(max_age_steps={max_age_steps})"
        )
        self._remove_indices(indices_to_remove)

    def _count_for_target(
        self, target_step: int, max_age_steps: int | None = None
    ) -> int:
        """Count trajectories usable for ``target_step``."""
        return sum(
            1
            for trajectory_version, target in zip(
                self.trajectory_versions, self.target_weight_versions
            )
            if target == target_step
            and self._is_valid_for_target(
                trajectory_version, target_step, max_age_steps
            )
        )

    def _truncate_to_max_size(self, current_training_step: int | None = None) -> None:
        """Truncate restored state to ``max_size`` after resume cleanup."""
        if len(self.trajectories) <= self.max_size:
            return

        print(
            f"Truncating restored buffer from {len(self.trajectories)} "
            f"to max_size={self.max_size}"
        )
        if current_training_step is None:
            indices_to_keep = list(
                range(len(self.trajectories) - self.max_size, len(self.trajectories))
            )
        else:
            prioritized_indices = sorted(
                range(len(self.trajectories)),
                key=lambda i: (self.target_weight_versions[i], i),
            )
            indices_to_keep = sorted(prioritized_indices[: self.max_size])

        self.trajectories = [self.trajectories[i] for i in indices_to_keep]
        self.trajectory_versions = [
            self.trajectory_versions[i] for i in indices_to_keep
        ]
        self.target_weight_versions = [
            self.target_weight_versions[i] for i in indices_to_keep
        ]

    def get_trajectories_needed(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> int:
        """Return additional trajectories needed for ``target_step``."""
        with self._lock:
            current_count = self._count_for_target(target_step, max_age_steps)
            return max(0, num_prompts_per_step - current_count)

    def has_complete_batch(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> bool:
        """Return whether ``target_step`` has enough trajectories to train."""
        with self._lock:
            current_count = self._count_for_target(target_step, max_age_steps)
            return current_count >= num_prompts_per_step

    def _remove_incomplete_target_steps(self, num_prompts_per_step: int) -> None:
        """Remove target steps without a complete batch."""
        target_counts = Counter(self.target_weight_versions)
        incomplete_targets = {
            target
            for target, count in target_counts.items()
            if count < num_prompts_per_step
        }
        if not incomplete_targets:
            print(f"   All target steps have complete batches ({num_prompts_per_step})")
            return

        print(f"   Removing incomplete target steps: {sorted(incomplete_targets)}")
        original_count = len(self.trajectories)
        indices_to_keep = [
            i
            for i, target in enumerate(self.target_weight_versions)
            if target not in incomplete_targets
        ]
        self.trajectories = [self.trajectories[i] for i in indices_to_keep]
        self.trajectory_versions = [
            self.trajectory_versions[i] for i in indices_to_keep
        ]
        self.target_weight_versions = [
            self.target_weight_versions[i] for i in indices_to_keep
        ]
        print(
            f"   Removed {original_count - len(self.trajectories)} trajectories "
            "from incomplete target steps"
        )

        if self.target_weight_versions:
            first_remaining_target = min(self.target_weight_versions)
            self.last_target_weight_already_generated = min(
                self.last_target_weight_already_generated,
                first_remaining_target - 1,
            )
        else:
            self.last_target_weight_already_generated = -1


@ray.remote  # pragma: no cover
class ReplayBuffer(ReplayBufferImpl):
    pass


@ray.remote  # pragma: no cover
class AsyncTrajectoryCollector:
    """Collects trajectories asynchronously and adds them to replay buffer."""

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        master_config: MasterConfig,
        replay_buffer: Any,
        start_step: int = 0,
    ):
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config
        self.replay_buffer = replay_buffer
        self.running = False

        self._pg_lock: _threading.Lock = _threading.Lock()

        # Event for manual pause/resume control
        self._manual_pause_cleared = _threading.Event()
        self._manual_pause_cleared.set()

        self._refit_pause_cleared = _threading.Event()
        self._refit_pause_cleared.set()  # Start in cleared state

        self.current_weight_version: int = start_step
        self.initial_weight_version: int = start_step

        # Track when generation limits cause collection to pause
        self._last_limit_warning_version = None

        # Event to signal when generation limits are cleared (more efficient than polling)
        self._generation_limit_cleared = _threading.Event()
        self._generation_limit_cleared.set()  # Start in cleared state

        # Track threads
        self._inflight_threads: set[_threading.Thread] = set()
        self._threads_lock: _threading.Lock = _threading.Lock()

        # Limit in-flight generator requests to num_prompts_per_step * max_trajectory_age_steps
        # This value limits the parallelism of the generation requests.
        max_inflight = (
            int(self.master_config["grpo"]["num_prompts_per_step"])
            * int(self.master_config["grpo"]["async_grpo"]["max_trajectory_age_steps"])
        ) or 1
        self._inflight_sema = _threading.Semaphore(max_inflight)

        # Simple lock to prevent race conditions when checking/spawning workers
        self._generation_check_lock: _threading.Lock = _threading.Lock()
        # Track which target weights are currently being generated (globally)
        self._generating_targets: set[int] = set()
        # Track spawned, buffered, and completed prompt-group workers per target.
        self._spawned_per_target: dict[int, int] = {}
        self._buffered_per_target: dict[int, int] = {}
        self._completed_per_target: dict[int, int] = {}
        self._spawning_targets: set[int] = set()
        self._counter_lock: _threading.Lock = _threading.Lock()

    def _calculate_target_weights(self, generation_weight_version: int) -> list[int]:
        """Calculate target weight versions for given generation weight version.

        The list of versions returned enumerate the possible version a generation
        server can target. These versions are looped over to see what training
        step they can target. If all target versions are exhausted, this generation
        server will remain idle until the next weight update.

        Example:
        generation_weight_version = 10
        max_trajectory_age_steps = 4

        Returns:
            [11, 12, 13, 14]  # Meaning this generation server can create trajectories for training step 11, 12, 13, 14
        """
        # Read async config strictly from grpo.async_grpo
        async_cfg = self.master_config.get("grpo", {}).get("async_grpo", {})
        max_trajectory_age = async_cfg["max_trajectory_age_steps"]
        if generation_weight_version == self.initial_weight_version:
            return [
                i
                for i in range(
                    self.initial_weight_version,
                    self.initial_weight_version + max_trajectory_age + 1,
                )
            ]

        return [generation_weight_version + i for i in range(1, max_trajectory_age + 1)]

    def _get_next_target_for_generation(
        self, generation_weight_version: int
    ) -> Optional[int]:
        """Get the next target weight that needs generation (if any)."""
        target_weights = self._calculate_target_weights(generation_weight_version)
        num_prompts = int(self.master_config["grpo"]["num_prompts_per_step"])
        max_age_steps = int(
            self.master_config["grpo"]["async_grpo"]["max_trajectory_age_steps"]
        )
        last_consumed_target = ray.get(
            self.replay_buffer.get_last_target_weight_already_generated.remote()
        )

        with self._generation_check_lock:
            for target_weight in target_weights:
                if target_weight <= last_consumed_target:
                    continue
                if target_weight in self._generating_targets:
                    continue

                trajectories_needed = ray.get(
                    self.replay_buffer.get_trajectories_needed.remote(
                        target_weight, num_prompts, max_age_steps
                    )
                )
                if trajectories_needed <= 0:
                    continue

                self._generating_targets.add(target_weight)
                if trajectories_needed < num_prompts:
                    print(
                        f"🎯 Reserved target weight {target_weight} for gap-filling "
                        f"(need {trajectories_needed}/{num_prompts} more trajectories)"
                    )
                else:
                    print(f"🎯 Reserved target weight {target_weight} for generation")
                return target_weight

        return None

    def set_weight_version(self, version: int) -> None:
        self.current_weight_version = version

        # Resume collection if it was paused due to generation limits
        was_paused = not self._generation_limit_cleared.is_set()
        if was_paused:
            self._generation_limit_cleared.set()  # Signal that collection can resume
            print(f"🔄 Updated weight version to {version}, resuming collection")
        else:
            print(f"🔄 Updated weight version to {version}")

    def _should_pause_for_generation_limits(self) -> bool:
        """Check if collection should be paused due to generation limits."""
        try:
            target_weights = self._calculate_target_weights(self.current_weight_version)
            num_prompts = int(self.master_config["grpo"]["num_prompts_per_step"])
            max_age_steps = int(
                self.master_config["grpo"]["async_grpo"]["max_trajectory_age_steps"]
            )
            last_consumed_target = ray.get(
                self.replay_buffer.get_last_target_weight_already_generated.remote()
            )

            # Check if any target weight in our range needs generation
            with self._generation_check_lock:
                for target_weight in target_weights:
                    if target_weight <= last_consumed_target:
                        continue
                    if target_weight in self._generating_targets:
                        continue
                    trajectories_needed = ray.get(
                        self.replay_buffer.get_trajectories_needed.remote(
                            target_weight, num_prompts, max_age_steps
                        )
                    )
                    if trajectories_needed > 0:
                        return False  # Found a target that needs generation

            print(
                f"⏸️ All target weights {target_weights} already generated or in progress, pausing"
            )
            return True
        except Exception:
            return False

    def start_collection(self, dataloader: StatefulDataLoader) -> None:
        """Start collecting trajectories from dataloader."""
        self.running = True
        self.dataloader = dataloader

        print("Started continuous trajectory collection")

        self.collection_thread = _threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        print("Collection thread started, start_collection returning")

    def _collection_loop(self):
        """Run the collection loop in background thread."""
        try:
            for batch in self.dataloader:
                if not self.running:
                    break

                # Check if manually paused and wait
                if not self._manual_pause_cleared.is_set() and self.running:
                    self._manual_pause_cleared.wait()

                # Check if refit is in progress and wait
                if not self._refit_pause_cleared.is_set() and self.running:
                    print("⏸️ Pausing collection for refit...")
                    self._refit_pause_cleared.wait()
                    print("▶️ Refit completed, resuming collection")

                # Check if generation limits require pausing collection
                if self._should_pause_for_generation_limits() and self.running:
                    # Only log warning once per weight version
                    if self._last_limit_warning_version != self.current_weight_version:
                        async_cfg = self.master_config.get("grpo", {}).get(
                            "async_grpo", {}
                        )
                        max_trajectory_age = async_cfg["max_trajectory_age_steps"]
                        target_weights = [
                            self.current_weight_version + i
                            for i in range(max_trajectory_age)
                        ]

                        print(
                            f"⏸️ Pausing collection: all target weights {target_weights} for weight version {self.current_weight_version} "
                            f"already exist in buffer. Waiting for weight update..."
                        )
                        self._last_limit_warning_version = self.current_weight_version

                        self._generation_limit_cleared.clear()  # Clear the event to pause

                    # Efficiently wait for generation limits to be cleared (no polling!)
                    self._generation_limit_cleared.wait()

                    # Double-check we're still running after being woken up
                    if not self.running:
                        break

                if not self.running:
                    break

                self._process_batch(batch)

        except Exception as e:
            print(f"❌ Error in trajectory collection: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.running = False
            print("🛑 Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Process a single batch and generate for one target weight."""
        target_weight: Optional[int] = None
        try:
            generation_weight_version = self.current_weight_version
            num_generations = self.master_config["grpo"]["num_generations_per_prompt"]
            num_prompts_in_batch = batch.size
            num_prompts_per_step = int(
                self.master_config["grpo"]["num_prompts_per_step"]
            )
            max_age_steps = int(
                self.master_config["grpo"]["async_grpo"]["max_trajectory_age_steps"]
            )

            # Get the next target weight that needs generation
            target_weight = self._get_next_target_for_generation(
                generation_weight_version
            )

            if target_weight is None:
                print(
                    f"🔄 No targets need generation for weight {generation_weight_version}"
                )
                return

            print(
                f"🎯 Generating for target weight {target_weight} from generation_weight_version {generation_weight_version}"
            )

            trajectories_needed = ray.get(
                self.replay_buffer.get_trajectories_needed.remote(
                    target_weight, num_prompts_per_step, max_age_steps
                )
            )
            num_prompts_to_generate = min(num_prompts_in_batch, trajectories_needed)
            if num_prompts_to_generate == 0:
                print(
                    f"🔄 Target {target_weight} already has enough trajectories, skipping"
                )
                with self._generation_check_lock:
                    self._generating_targets.discard(target_weight)
                return

            if num_prompts_to_generate < num_prompts_in_batch:
                print(
                    f"🎯 Gap-filling for target weight {target_weight}: "
                    f"generating {num_prompts_to_generate}/{num_prompts_in_batch} "
                    f"prompts (need {trajectories_needed} more trajectories)"
                )

            # Generate only the prompt groups needed for this target. While the
            # spawn loop is open, workers may finish before later workers start,
            # so reservation release is deferred until spawning closes.
            started = 0
            with self._counter_lock:
                self._spawning_targets.add(target_weight)
            try:
                for prompt_idx in range(num_prompts_to_generate):
                    # Wait for refit to complete if in progress
                    if not self._refit_pause_cleared.is_set() and self.running:
                        with self._threads_lock:
                            active_threads = len(self._inflight_threads)
                        print(
                            f"⏸️ Waiting for refit to complete before starting new generation ({active_threads} threads still active)"
                        )
                        print(
                            "   Note: With vLLM V1 async engine, active threads can complete during weight update"
                        )
                        self._refit_pause_cleared.wait()

                        # After refit finishes if weight version has updated, reflect that in the new trajectories
                        generation_weight_version = self.current_weight_version

                    single_prompt_batch = batch.slice(prompt_idx, prompt_idx + 1)
                    repeated_batch = single_prompt_batch.repeat_interleave(
                        num_generations
                    )

                    worker = _threading.Thread(
                        target=self._run_prompt_group_worker,
                        args=(
                            repeated_batch,
                            generation_weight_version,
                            target_weight,
                            prompt_idx,
                        ),
                        daemon=True,
                    )
                    self._inflight_sema.acquire()
                    registered = False
                    try:
                        with self._threads_lock:
                            self._inflight_threads.add(worker)
                        with self._counter_lock:
                            self._spawned_per_target[target_weight] = (
                                self._spawned_per_target.get(target_weight, 0) + 1
                            )
                            spawned_count = self._spawned_per_target[target_weight]
                        registered = True
                        worker.start()
                    except Exception:
                        # The worker never ran, so it won't release its slot or
                        # run its finally block; undo the bookkeeping here.
                        with self._threads_lock:
                            self._inflight_threads.discard(worker)
                        if registered:
                            with self._counter_lock:
                                updated_count = (
                                    self._spawned_per_target.get(target_weight, 0) - 1
                                )
                                if updated_count > 0:
                                    self._spawned_per_target[target_weight] = (
                                        updated_count
                                    )
                                else:
                                    self._spawned_per_target.pop(target_weight, None)
                        self._inflight_sema.release()
                        raise
                    started += 1
                    print(
                        f"📊 Started worker {started}/{num_prompts_to_generate} for "
                        f"target_weight={target_weight} ({spawned_count} total)"
                    )
            finally:
                if started < num_prompts_to_generate:
                    print(
                        f"⚠️ Only {started}/{num_prompts_to_generate} workers "
                        f"started for target_weight={target_weight}"
                    )
                with self._counter_lock:
                    self._spawning_targets.discard(target_weight)
                self._maybe_release_target(target_weight)

            self._cleanup_finished_threads()

        except Exception as e:
            if target_weight is not None:
                with self._counter_lock:
                    self._spawning_targets.discard(target_weight)
                self._maybe_release_target(target_weight)
            print(f"❌ Error processing batch: {e}")
            import traceback

            traceback.print_exc()

    def get_weight_version(self) -> int:
        return self.current_weight_version

    def pause(self) -> None:
        """Pause trajectory collection."""
        self._manual_pause_cleared.clear()  # Signal collection to pause
        print("Trajectory collection paused")

    def resume(self) -> None:
        """Resume trajectory collection."""
        self._manual_pause_cleared.set()  # Signal collection to resume
        print("Trajectory collection resumed")

    def prepare_for_refit(self) -> None:
        """Pause new generation starts and optionally wait for pending generations.

        For vLLM V1 async engine, leverages in-flight weight updates via collective_rpc,
        allowing ongoing generations to continue with their current KV caches while
        weights are updated. This significantly improves async performance.

        For non-async engines, waits for all pending generations to complete before refit.
        """
        start_time = time.time()
        print("🔄 Preparing for refit: pausing new generations...")

        # Pause new generation starts
        self._refit_pause_cleared.clear()
        print("⏸️ New generation starts paused")

        # Check if we're using vLLM async engine
        vllm_cfg = (
            self.master_config.get("policy", {})
            .get("generation", {})
            .get("vllm_cfg", {})
        )
        is_async_engine = vllm_cfg.get("async_engine", False)
        in_flight_weight_updates = (
            self.master_config.get("grpo", {})
            .get("async_grpo", {})
            .get("in_flight_weight_updates", False)
        )

        if is_async_engine and in_flight_weight_updates:
            # vLLM V1 async engine supports in-flight weight updates
            # Ongoing generations will continue with their current KV caches
            # New generations (after weight update) will use the updated weights
            print(
                "🚀 Using vLLM V1 in-flight weight update - skipping wait for pending generations"
            )
            print(
                f"   {len(self._inflight_threads)} ongoing generations will complete with current weights"
            )
        else:
            # For non-async engines, wait for all pending generations to complete
            print(
                "⏸️ Non-async engine: waiting for all pending generations to complete..."
            )
            self.wait_for_pending_generations()

        elapsed = time.time() - start_time
        print(f"✅ Ready for refit (took {elapsed:.2f}s)")

    def resume_after_refit(self) -> None:
        """Resume new generation starts after refit is complete."""
        print("🔄 Resuming generation starts after refit")

        # Invalidate&recompute vLLM caches after the in-flight weight updates if
        # recompute_kv_cache_after_weight_updates is True (AREAL-style implementation).
        # Otherwise, keep using the stale KV caches (Magistral-style implementation).
        async_cfg = self.master_config.get("grpo", {}).get("async_grpo", {})
        if async_cfg.get("in_flight_weight_updates", False) and async_cfg.get(
            "recompute_kv_cache_after_weight_updates", False
        ):
            try:
                print("🔄 Invalidating vLLM prefix/KV caches after weight update")
                invalidated = self.policy_generation.invalidate_kv_cache()
                if invalidated:
                    print("✅ Invalidated vLLM prefix/KV caches after weight update")
                else:
                    print(
                        "⚠️ vLLM cache invalidation reported partial/unsuccessful on some workers"
                    )
            except Exception as e:
                print(f"⚠️ Failed to invalidate vLLM caches: {e}")

        self._refit_pause_cleared.set()

    def wait_for_pending_generations(self) -> None:
        """Wait for all in-flight generation threads to complete."""
        start_time = time.time()

        while True:
            with self._threads_lock:
                finished = {t for t in self._inflight_threads if not t.is_alive()}
                for t in finished:
                    self._inflight_threads.remove(t)

                pending_count = len(self._inflight_threads)

            if pending_count == 0:
                print("✅ All generation threads completed")
                break

            elapsed = time.time() - start_time
            print(
                f"⏳ Waiting for {pending_count} pending generation threads... ({elapsed:.1f}s elapsed)"
            )
            time.sleep(0.5)

    def get_dataloader_state(self) -> dict:
        """Get the current dataloader state for checkpointing."""
        if hasattr(self, "dataloader") and hasattr(self.dataloader, "state_dict"):
            return self.dataloader.state_dict()
        return {}

    def _cleanup_finished_threads(self) -> None:
        with self._threads_lock:
            finished = {t for t in self._inflight_threads if not t.is_alive()}
            for t in finished:
                self._inflight_threads.remove(t)

    def _maybe_release_target(self, target_weight_version: int) -> None:
        """Release a target's reservation once all its workers have completed.

        A worker counts as "completed" whether or not it managed to buffer a
        trajectory. The reservation is released exactly when the number of
        completed workers reaches the number spawned for the target and no more
        workers are being spawned for the same target. Safe to call repeatedly:
        the reservation is discarded at most once and the per-target counters
        are dropped on release (so a later re-reservation starts from a clean
        slate and the dicts don't grow unbounded).
        """
        with self._counter_lock:
            if target_weight_version in self._spawning_targets:
                return
            completed = self._completed_per_target.get(target_weight_version, 0)
            spawned = self._spawned_per_target.get(target_weight_version, 0)
            if completed < spawned:
                return
            buffered = self._buffered_per_target.get(target_weight_version, 0)
            self._spawned_per_target.pop(target_weight_version, None)
            self._buffered_per_target.pop(target_weight_version, None)
            self._completed_per_target.pop(target_weight_version, None)
            self._spawning_targets.discard(target_weight_version)

        with self._generation_check_lock:
            if target_weight_version in self._generating_targets:
                self._generating_targets.discard(target_weight_version)
                print(
                    "🧹 Released reservation for target weight "
                    f"{target_weight_version} ({spawned} workers completed, "
                    f"{buffered} buffered)"
                )

    def _run_prompt_group_worker(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: int,
        prompt_idx: int,
    ) -> None:
        try:
            # Import here to avoid circular dependency
            from nemo_rl.algorithms.grpo import _should_use_nemo_gym
            from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout

            # Run rollout for this prompt group
            # Async engine supports concurrent generation; avoid locking
            # Check if we should use nemo_gym (similar to synchronous GRPO)
            if _should_use_nemo_gym(self.master_config):
                generation_config = self.master_config["policy"]["generation"]
                env_cfg = self.master_config.get("env") or {}
                nemo_gym_rollout_result = run_async_nemo_gym_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    max_seq_len=None,
                    generation_config=generation_config,
                    max_rollout_turns=None,
                    greedy=False,
                )
                final_batch = nemo_gym_rollout_result.final_batch
                rollout_metrics = nemo_gym_rollout_result.rollout_metrics
            else:
                final_batch, rollout_metrics = run_async_multi_turn_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    max_seq_len=self.master_config["policy"][
                        "max_total_sequence_length"
                    ],
                    max_rollout_turns=self.master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                )

            # Move to CPU and push to buffer (avoid blocking on GC/push)
            final_batch_cpu = final_batch.to("cpu")
            del final_batch

            trajectory_group = {
                "batch": final_batch_cpu,
                "rollout_metrics": rollout_metrics,
                "timestamp": time.time(),
            }

            # Use exponential backoff when buffer is full
            try:
                backoff_delay = 0.01
                while self.running:
                    status = ray.get(
                        self.replay_buffer.add.remote(
                            trajectory_group,
                            generation_weight_version,
                            target_weight_version,
                        )
                    )
                    if status == "success":
                        with self._counter_lock:
                            self._buffered_per_target[target_weight_version] = (
                                self._buffered_per_target.get(target_weight_version, 0)
                                + 1
                            )
                            buffered_count = self._buffered_per_target[
                                target_weight_version
                            ]
                            spawned_count = self._spawned_per_target.get(
                                target_weight_version, 0
                            )
                        print(
                            f"📦 Buffered per-prompt group (prompt_idx {prompt_idx}, "
                            f"target_weight {target_weight_version}) "
                            f"[{buffered_count}/{spawned_count} buffered]"
                        )
                        break
                    elif status == "full":
                        # Exponential backoff up to 0.5 second
                        time.sleep(min(backoff_delay, 0.5))
                        backoff_delay *= 1.5
                    else:
                        # Unexpected status, wait briefly
                        time.sleep(0.01)
            except Exception as e:
                print(f"❌ Failed to enqueue per-prompt group to buffer: {e}")
                import traceback

                traceback.print_exc()
        except Exception as e:
            print(f"❌ Error in prompt group worker: {e}")
            import traceback

            traceback.print_exc()
        finally:
            with self._counter_lock:
                self._completed_per_target[target_weight_version] = (
                    self._completed_per_target.get(target_weight_version, 0) + 1
                )
            self._maybe_release_target(target_weight_version)

            # Detach thread record when finished
            with self._threads_lock:
                current = _threading.current_thread()
                if current in self._inflight_threads:
                    self._inflight_threads.remove(current)
            try:
                self._inflight_sema.release()
            except Exception:
                import traceback

                traceback.print_exc()
