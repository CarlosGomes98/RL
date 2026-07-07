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

import os
from importlib.util import find_spec


_TE_DPA_UTILS_PATH = "pytorch/attention/dot_product_attention/utils.py"
_TE_FA2_ARCH_ALLOWLIST = "((8, 0), (9, 0), (10, 0), (12, 0))"
_TE_FA2_ARCH_ALLOWLIST_SM103 = "((8, 0), (9, 0), (10, 0), (10, 3), (12, 0))"


def _get_transformer_engine_file(relative_path: str) -> str:
    """Return absolute path to a Transformer Engine file or raise if it cannot be found.

    The relative_path should be a POSIX-style path under the transformer_engine
    package root, e.g. "pytorch/triton/permutation.py".
    """
    spec = find_spec("transformer_engine")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "Transformer Engine package not found while attempting to patch "
            f"'{relative_path}'. Ensure `transformer-engine` is installed and "
            "available in this environment."
        )

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))

    if not os.path.exists(file_path):
        raise RuntimeError(
            "Failed to locate expected Transformer Engine file to patch. "
            f"Looked for '{relative_path}' at '{file_path}'. "
            "This likely indicates an unexpected Transformer Engine installation "
            "layout or version mismatch."
        )

    return file_path


def apply_transformer_engine_patch():
    """Apply patch from https://github.com/NVIDIA/TransformerEngine/pull/2286/files.

    This locates the target file via importlib metadata instead of importing
    `transformer_engine`, to avoid side effects during initialization. If the
    permutation module has already been imported, it will be reloaded so that
    the patched source takes effect.
    """
    try:
        perm_file = _get_transformer_engine_file("pytorch/triton/permutation.py")

        with open(perm_file, "r") as f:
            content = f.read()

        if "get_int_dtype = triton.constexpr_function(get_int_dtype)" not in content:
            print(f"Applying Triton fix to {perm_file}...")

            # 1. Replace the usage
            old_usage = "idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)"
            new_usage = "idtype = get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)"

            # 2. Insert the definition before the first @triton.jit
            jit_anchor = "@triton.jit"

            new_definition = (
                "\n\n"
                "get_int_dtype = core.get_int_dtype\n"
                "get_int_dtype = triton.constexpr_function(get_int_dtype)\n"
            )

            new_content = None
            if old_usage in content:
                temp_content = content.replace(old_usage, new_usage)

                if jit_anchor in temp_content:
                    new_content = temp_content.replace(
                        jit_anchor, new_definition + jit_anchor, 1
                    )

            if new_content:
                try:
                    with open(perm_file, "w") as f:
                        f.write(new_content)
                    print("Successfully patched transformer_engine permutation.py.")
                except OSError as e:
                    print(
                        f"Could not write patch to transformer_engine (permission denied?): {e}"
                    )

        # If the permutation module is already imported in this process,
        # reload it so that the patched source takes effect for subsequent use.
        import importlib
        import sys

        perm_module_name = "transformer_engine.pytorch.triton.permutation"
        if perm_module_name in sys.modules:
            importlib.reload(sys.modules[perm_module_name])

    except Exception as e:
        print(f"Error checking/patching transformer_engine: {e}")


def apply_transformer_engine_sm103_flash_attention_patch() -> None:
    """Allow FlashAttention-2 with a 256-wide head on GB300 in TE 2.15."""
    import fcntl

    dpa_utils_file = _get_transformer_engine_file(_TE_DPA_UTILS_PATH)
    with open(dpa_utils_file, "r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        content = file.read()

        if _TE_FA2_ARCH_ALLOWLIST_SM103 not in content:
            occurrence_count = content.count(_TE_FA2_ARCH_ALLOWLIST)
            if occurrence_count != 1:
                raise RuntimeError(
                    "Could not identify TransformerEngine's pinned FlashAttention-2 "
                    "architecture allowlist exactly once; refusing a fuzzy sm10.3 "
                    f"patch. Found {occurrence_count} matches in {dpa_utils_file}."
                )

            content = content.replace(
                _TE_FA2_ARCH_ALLOWLIST,
                _TE_FA2_ARCH_ALLOWLIST_SM103,
                1,
            )
            file.seek(0)
            file.write(content)
            file.truncate()
            file.flush()
            print(
                "Enabled TransformerEngine FlashAttention-2 on sm10.3: "
                f"{dpa_utils_file}"
            )

    import importlib
    import sys

    module_name = "transformer_engine.pytorch.attention.dot_product_attention.utils"
    if module_name in sys.modules:
        module = sys.modules[module_name]
        flash_attention_utils = module.FlashAttentionUtils
        reloaded_module = importlib.reload(module)
        # backends.py records package detection on this class at import time.
        # Keep that state instead of replacing it with the reloaded defaults.
        reloaded_module.FlashAttentionUtils = flash_attention_utils
