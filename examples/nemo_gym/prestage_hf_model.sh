#!/usr/bin/env bash
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

# Pre-stage an HF model into a run's HF cache by SYMLINK from an already-populated
# cache, so megatron setup's AutoBridge.from_hf_pretrained(model_name) finds every
# file locally instead of snapshot_download-ing the full repo (~470 GB for 235B) and
# triggering an HTTP 429 storm across all policy workers at startup.
#
# This copies NO weight bytes -- it mirrors the source cache's blob/snapshot/ref
# layout with symlinks (blobs point at the source's read-only files; refs/snapshots
# are local and writable). Idempotent: re-running only fills in anything missing.
#
# Usage (defaults target this recipe + a known-good source cache on this cluster):
#   bash examples/nemo_gym/prestage_hf_model.sh
#   MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507 \
#   SRC_HUB=/lustre/.../some/hf_hub/hub \
#   HF_HOME=$PWD/.cache \
#     bash examples/nemo_gym/prestage_hf_model.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
# A populated HF hub cache dir (the one that contains the "models--*" subdirs).
SRC_HUB="${SRC_HUB:-/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/lukaszp/hf_hub/hub}"
# Must match HF_HOME used at launch (the launcher uses $PWD/.cache).
HF_HOME="${HF_HOME:-$PWD/.cache}"

repo_dir="models--${MODEL//\//--}"
SRC="$SRC_HUB/$repo_dir"
DST="$HF_HOME/hub/$repo_dir"

if [[ ! -d "$SRC/snapshots" ]]; then
  echo "ERROR: source cache not found or incomplete: $SRC" >&2
  echo "       set SRC_HUB to a hub cache dir containing $repo_dir" >&2
  exit 1
fi

echo "Pre-staging $MODEL"
echo "  from: $SRC"
echo "  to:   $DST"

mkdir -p "$DST/blobs" "$DST/snapshots" "$DST/refs"

# 1. blobs: symlink each content-addressed blob to the source (read-only is fine).
for blob in "$SRC"/blobs/*; do
  name="$(basename "$blob")"
  [[ -e "$DST/blobs/$name" ]] || ln -s "$blob" "$DST/blobs/$name"
done

# 2. snapshots: recreate each revision tree, preserving the relative "../../blobs/<sha>"
#    symlink targets so they resolve within DST.
for rev_dir in "$SRC"/snapshots/*; do
  rev="$(basename "$rev_dir")"
  mkdir -p "$DST/snapshots/$rev"
  while IFS= read -r -d '' f; do
    rel="$(basename "$f")"
    [[ -e "$DST/snapshots/$rev/$rel" ]] && continue
    if [[ -L "$f" ]]; then
      ln -s "$(readlink "$f")" "$DST/snapshots/$rev/$rel"
    else
      ln -s "$f" "$DST/snapshots/$rev/$rel"
    fi
  done < <(find "$rev_dir" -maxdepth 1 -mindepth 1 -print0)
done

# 3. refs: small text files (e.g. main -> <revision>); copy so they are local/writable.
if [[ -d "$SRC/refs" ]]; then
  cp -n "$SRC"/refs/* "$DST/refs/" 2>/dev/null || true
fi

echo "Done. Staged $(find "$DST/blobs" -maxdepth 1 | wc -l) blob links under $DST"
echo "Verify with: HF_HOME=$HF_HOME python -c \"from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', local_files_only=True))\""
