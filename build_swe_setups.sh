#!/bin/bash
# Populate the swe_agents helper setups. The container has /opt/.../swe_*_setup as
# symlinks -> shared Lustre; we mkdir the symlink TARGETS (on Lustre) so the setup
# scripts can cd into them and build there. No --container-save needed (data -> Lustre).
set -x
SWE=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents
cd "$SWE" || { echo "FATAL: no $SWE"; exit 1; }

# R2E harness fix (ephemeral in this session, but the R2E venv it builds on Lustre is what counts)
sed -i 's#^\(uv pip install -p \$r2e_gym_dir/venv/bin/python -e \. --no-cache\)#\1\nuv pip install -p $r2e_gym_dir/venv/bin/python --reinstall cffi cryptography#' setup_scripts/r2e_gym.sh
grep -n "reinstall cffi" setup_scripts/r2e_gym.sh || echo "WARN: r2e sed did not match"

# mkdir the symlink target (Lustre) if $1 is a symlink, else mkdir the dir itself
ensure_dir() {
  if [ -L "$1" ]; then
    local t; t="$(readlink "$1")"
    echo "SYMLINK $1 -> $t"
    mkdir -p "$t"
  else
    mkdir -p "$1"
  fi
}

S="$SWE/swe_openhands_setup";             ensure_dir "$S"; SETUP_DIR="$S" MINIFORGE_DIR="$S/miniforge3" OPENHANDS_DIR="$S/OpenHands" AGENT_FRAMEWORK_REPO=https://github.com/sdevare-nv/nv-OpenHands.git AGENT_FRAMEWORK_COMMIT=0d766ad06b2be64a42e6f0175b9ebcc4a06599d9 bash setup_scripts/openhands.sh
S="$SWE/swe_r2e_gym_setup";               ensure_dir "$S"; SETUP_DIR="$S" UV_DIR="$S/uv" PYTHON_DIR="$S/python" R2E_GYM_DIR="$S/R2E-Gym" EVAL_HARNESS_REPO=https://github.com/sdevare-nv/nv-R2E-Gym.git EVAL_HARNESS_COMMIT=local-eval bash setup_scripts/r2e_gym.sh
S="$SWE/swe_swebench_setup";              ensure_dir "$S"; SETUP_DIR="$S" UV_DIR="$S/uv" PYTHON_DIR="$S/python" SWEBENCH_DIR="$S/SWE-bench" SWEBENCH_REPO=https://github.com/HeyyyyyyG/SWE-bench.git SWEBENCH_COMMIT=HEAD bash setup_scripts/swebench.sh
S="$SWE/swe_swebench_multilingual_setup"; ensure_dir "$S"; SETUP_DIR="$S" UV_DIR="$S/uv" PYTHON_DIR="$S/python" SWEBENCH_DIR="$S/SWE-bench_Multilingual" SWEBENCH_REPO=https://github.com/Kipok/SWE-bench.git SWEBENCH_COMMIT=HEAD bash setup_scripts/swebench_multilingual.sh
S="$SWE/swe_rebench_setup";               ensure_dir "$S"; SETUP_DIR="$S" REBENCH_DIR="$S/SWE-rebench-V2" bash setup_scripts/swe_rebench.sh

echo "============ VERIFY ============"
ok=1
"$SWE/.venv/bin/python" -c "import ray; print('venv ray', ray.__version__)" || { echo "FAIL: venv ray"; ok=0; }
ls -lL "$SWE/swe_openhands_setup/OpenHands/.venv/bin/python" || { echo "FAIL: openhands"; ok=0; }
"$SWE/swe_r2e_gym_setup/R2E-Gym/venv/bin/python" -c "import r2egym, kubernetes; print('r2e OK')" || { echo "FAIL: r2e/kubernetes"; ok=0; }
ls -ldL "$SWE/swe_swebench_setup/SWE-bench" || { echo "FAIL: swebench"; ok=0; }
ls -ldL "$SWE/swe_swebench_multilingual_setup/SWE-bench_Multilingual" || { echo "FAIL: swebench_ml"; ok=0; }
ls -lL "$SWE/swe_rebench_setup/SWE-rebench-V2/agent/log_parsers.py" || { echo "FAIL: rebench"; ok=0; }
[ "$ok" = 1 ] && echo "============ BUILD OK ============" || echo "============ BUILD HAD FAILURES ============"
