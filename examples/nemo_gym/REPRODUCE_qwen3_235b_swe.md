# Reproducing the GRPO Qwen3-235B SWE-OpenHands run

This run has **zero container mutations** — the container image is used read-only; every
heavy artifact lives on Lustre, and every fix is either a repo file or a launch-time env var.
So "make it run on a fresh node" = make sure the persistent artifacts exist + use the launch
command below. Nothing needs `--container-save`.

## 1. What actually has to exist (persistent, one-time)

| Artifact | Path (this setup) | How to (re)create |
|---|---|---|
| Container image | `/lustre/fsw/.../mfutrega/containers/nemo-rl.sqsh` | Prebuilt image. It has the NeMo-Gym code at `/opt/nemo-rl/...` **and** the `responses_api_agents/swe_agents/swe_*_setup` symlinks baked in, pointing at the absolute path `<REPO>/swe_setups/...`. (pyxis reads the sqsh host-side, so it may stay on `/lustre/fsw`.) |
| Repo, at a fixed path | `/lustre/fs1/.../mfutrega/code/rl` | The baked symlinks are absolute → keep the repo at this path, or rebuild the image with new symlink targets. |
| HF model staged in HF_HOME | `<REPO>/.cache/hub/models--Qwen--Qwen3-235B-...` (118 shards + tokenizer + config) | `bash examples/nemo_gym/prestage_hf_model.sh` (symlinks from a populated hub cache; copies no weight bytes). **Required** — see gotcha #5. |
| mcore checkpoint (HF→mcore) | `/lustre/fs1/.../mfutrega/mcore_ckpt_cache/Qwen/Qwen3-235B-A22B-Instruct-2507/iter_0000000` | One-time HF→mcore conversion; reused via `NRL_MEGATRON_CHECKPOINT_DIR`. Must be on a **mounted** mount (fs1), see gotcha #1. |
| SWE-agent setups | `<REPO>/swe_setups/{swe_openhands_setup,swe_r2e_gym_setup,...}` | Build **inside the container**: `bash build_swe_setups.sh` (+ `build_r2e_only.sh`/`build_r2e_targeted.sh`/`fix_r2e_hub.sh`/`fix_swebench.sh` for the R2E eval venv). Verify with `verify_swe_setups.sh` (see below). |
| R2E task images | `/lustre/fs1/.../lukaszp/data/sif/r2egym_*.sif` (4578) | Provided; `sif_dir` in the yaml points here. Must be on a mounted mount (fs1). |
| Datasets | `/lustre/fs1/.../cgomes/grpo-studies/data_swe/r2e_easy_l20_{train,val}.jsonl` | Provided. |

Verify the in-container setups are healthy (host-side `ls -lL` LIES — venv pythons are
symlinks to `/opt/...` that only resolve in-container):
```bash
srun --account=coreai_mlperf_training --partition=batch --nodes=1 --gres=gpu:1 --time=0:30:0 \
  --container-image=$CONTAINER --container-mounts=/lustre/fs1:/lustre/fs1 --no-container-mount-home \
  --container-workdir=$PWD bash verify_swe_setups.sh        # add a trailing FIX=1 env to repair R2E
```
Expected: OpenHands `.venv` OK, gym venv ray, and "R2E EVAL CHAIN OK".

## 2. Repo files that carry the fixes (commit these)

- `examples/nemo_gym/grpo_qwen3_235b_swe_openhands.yaml` — smoke config (incl. `sif_dir` on fs1, `moe_per_layer_logging: false`).
- `examples/nemo_gym/grpo_qwen3_235b_swe_openhands_learning.yaml` — learning config (bigger batch, val curve).
- `examples/nemo_gym/launch_nemo_gym_multinode_training.sh` — forwards `NRL_MEGATRON_CHECKPOINT_DIR`, `NEMO_GYM_SWE_WORKSPACE_ROOT`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; `--time=${SLURM_TIME:-1:0:0}`.
- helper scripts: `prestage_hf_model.sh`, `build_swe_setups.sh`, `build_r2e_*.sh`, `fix_*.sh`, `verify_swe_setups.sh`.

## 3. Canonical launch command

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/mfutrega/code/rl

SLURM_TIME=4:0:0 \                                  # walltime (omit -> 1h)
NEMO_GYM_SWE_WORKSPACE_ROOT=$PWD/swe_workspace \    # gotcha #2: MUST be shared-Lustre
REPO_LOCATION=$PWD \
EXP_NAME=grpo-qwen3-235b-swe-openhands-learn \
NUM_ACTOR_NODES=32 \                                # 16 -> DP1, 32 -> DP2
NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/mfutrega/mcore_ckpt_cache \  # gotcha #1: fs1, NOT fsw
CONTAINER_IMAGE_PATH=/lustre/fsw/portfolios/coreai/users/mfutrega/containers/nemo-rl.sqsh \
SLURM_ACCOUNT=coreai_mlperf_training SLURM_PARTITION=batch \
HF_TOKEN=<token> WANDB_API_KEY=null \
bash examples/nemo_gym/launch_nemo_gym_multinode_training.sh \
  --config examples/nemo_gym/grpo_qwen3_235b_swe_openhands_learning.yaml
```

## 4. The gotchas (each cost a run), now fixed

1. **Checkpoint dir on the wrong mount.** Only the repo's mount (`/lustre/fs1`) is bind-mounted; `/lustre/fsw` is invisible in-container. `NRL_MEGATRON_CHECKPOINT_DIR` is read in-container → use the **fs1** path or you get a multi-hour reconversion.
2. **`NEMO_GYM_SWE_WORKSPACE_ROOT` unset.** The gym writes per-task results there; runner actors are spread across nodes. If it's node-local (`/opt`), the server can't see them → `FileNotFoundError: .../apptainer_logs`. Point it at shared Lustre.
3. **SWE setups not built.** OpenHands container mount fails (`swe_openhands_setup/OpenHands doesn't exist`) → empty trajectory → `IndexError` in `apply_chat_template`. Build `swe_setups/` first.
4. **`moe_per_layer_logging` missing.** MoE models read `megatron_cfg.moe_per_layer_logging` unguarded in `train()` → `KeyError` on the first step. Set `false`.
5. **HF Hub 429 storm at scale.** Megatron tokenizer setup calls `model_info()` even when the model is cached. At 32 nodes (256 workers) this exceeds 1000 req/5min → `429` → `Unable to instantiate AutoTokenizer`. Fix: pre-stage the model in HF_HOME **and** set `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` (now in the launch script) so no Hub calls are made.
6. **`sif_dir` on `/lustre/fsw`.** Same mount issue as #1 — the agent's apptainer can't see fsw paths in-container. Point `sif_dir` at the fs1 target.
8. **Transient vLLM `cumem_allocator.cpp:119 CUDA Error: invalid argument` at startup** (→ `Engine core initialization failed`, often with a misleading follow-on CUDA OOM). This is a **bad-GPU/node** flake in vLLM sleep-mode allocator init, not config — it hit ~1 in N allocations. Find the offending worker IP in the driver log, map it (`getent hosts <ip>` → `pool0-NNNNN`), and resubmit excluding it: the launch script honors `SLURM_EXCLUDE=pool0-NNNNN` (passed to sbatch `--exclude`). Or just resubmit to get a fresh node set.

7. **`agent_max_turns` too high → empty-rollout crash.** If an agent trajectory's context exceeds `max_model_len` (80k) at some turn, vLLM rejects that request, the rollout comes back with no generation data, and `nemo_rl/environments/nemo_gym.py:_postprocess_*` raises `ValueError: NeMo Gym returned a result with no generation data` (its "prompt too long" hint is misleading — it reports the empty re-derived prompt length, e.g. "2 tokens"). One bad rollout kills the whole run; GRPO can't just drop it (needs exactly `num_generations` per prompt). The real cause is the agent filling context past `max_model_len`: `VLLMValidationError: You passed 81921 input tokens ... context length is only 81920`. Reducing `agent_max_turns` 20→15 did NOT fix it (still crashed at step 2). **Fix: size `max_total_sequence_length` above what a full `agent_max_turns` trajectory fills** — at 15 turns the agent reached ~80k, so 80k overflowed but **96k (98304)** gives headroom. (`max_model_len` tracks `max_total_sequence_length`; raising both also raises the training packing budget ~proportionally, so don't over-raise — 96k is +20% memory, 128k is +60%.) The OpenHands context cap (`oh_config.toml`/`max_input_tokens`) is baked in the container and not reachable via the recipe override, so sizing the seq length is the lever.

## 5. Watch the result
```bash
tensorboard --logdir results/<EXP_NAME>      # val:total_reward/mean, train:total_reward/mean
# rewards per step:  results/<EXP_NAME>/{val,train}_data_step*.jsonl
# raw trajectories:  swe_workspace/swebench_results_*/<instance>/{patch.diff,trajectories/.../output.jsonl,eval-outputs/*/report.json}
```
