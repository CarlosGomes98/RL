# Qwen 3.5 Newer-vLLM Runtime Overlays

These overlays are aligned to newer Gym/NeMo-RL images whose vLLM API has moved
past the v0.6 overlay assumptions, including the bleeding-edge Gym image built
from:

- NeMo-RL: `6b0a1c4d04557a8d40ceb9f23aba1777114ed7c3`
- Megatron-Bridge: `565b87b0c728a4b914cc3b61df5147751d2905bf`
- Megatron-LM: `a58373f332496f08c6584b3196233275ff69f175`

They are also used for the `bd8a540-55533441` NeMo-RL nightly Gym image.

They are mounted by `examples/nemo_gym/launch_nemo_gym_multinode_training.sh`
when `CONTAINER_IMAGE_PATH` contains `bleeding`, `nightly`, or the known
`bd8a540-55533441` nightly tag, unless `QWEN35_OVERLAY_DIR` or
`QWEN35_RUNTIME_OVERLAY` overrides that behavior.
