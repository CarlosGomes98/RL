#!/bin/bash
# In-container verification of the SWE-agent helper setups (OpenHands + R2E-Gym eval).
# Host-side checks are unreliable because venv/bin/python symlinks point at /opt/... paths
# that only resolve INSIDE the container. This script must run inside the NeMo-RL container.
# It is READ-ONLY by default; pass FIX=1 to apply the R2E eval-chain fixes if broken.
set -uo pipefail
SWE=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents
echo "########## SWE setup verification ##########"
echo "SWE dir: $SWE"
cd "$SWE" || { echo "FATAL: $SWE missing"; exit 1; }

echo "===== symlink targets ====="
for d in swe_openhands_setup swe_r2e_gym_setup swe_swebench_setup swe_swebench_multilingual_setup swe_rebench_setup; do
  printf "  %-34s -> " "$d"; readlink "$d" 2>/dev/null || echo "(not a symlink / missing)"
done

echo "===== gym venv ====="
"$SWE/.venv/bin/python" -c "import ray; print('  gym venv ray', ray.__version__)" 2>&1 | tail -2

echo "===== OpenHands ====="
ls -lL "$SWE/swe_openhands_setup/OpenHands/.venv/bin/python" >/dev/null 2>&1 \
  && echo "  OpenHands .venv python OK" || echo "  FAIL: OpenHands .venv python missing"
ls -dL "$SWE/swe_openhands_setup/miniforge3" >/dev/null 2>&1 \
  && echo "  miniforge3 OK" || echo "  FAIL: miniforge3 missing"

echo "===== R2E-Gym eval venv ====="
R2E="$SWE/swe_r2e_gym_setup/R2E-Gym"
PY="$R2E/venv/bin/python"
if ls -lL "$PY" >/dev/null 2>&1; then
  echo "  R2E venv python resolves OK"
  "$PY" -c "import huggingface_hub as h; print('  hub', h.__version__)" 2>&1 | tail -1
  "$PY" -c "from huggingface_hub import HfFolder; print('  HfFolder import OK')" 2>&1 | tail -1
  "$PY" -c "import swebench; from swebench.harness.log_parsers import get_eval_type; print('  swebench', swebench.__version__, 'get_eval_type OK')" 2>&1 | tail -1
  "$PY" -c "from r2egym.agenthub.runtime.local import LocalRuntime; from r2egym.agenthub.run import run_local_evaluation; import r2egym, kubernetes; print('  R2E EVAL CHAIN OK')" 2>&1 | tail -3
else
  echo "  FAIL: R2E venv python does not resolve in-container"
fi

if [ "${FIX:-0}" = "1" ]; then
  echo "########## FIX=1: applying R2E eval-chain fixes ##########"
  cd "$R2E" || exit 1
  export UV_INSTALL_DIR="$SWE/swe_r2e_gym_setup/uv"
  export UV_PYTHON_INSTALL_DIR="$SWE/swe_r2e_gym_setup/python"
  export PATH="$UV_INSTALL_DIR/bin:$PATH"
  command -v uv && uv --version

  # 1) import guards in utils.py (hub/transformers optional for local eval)
  python3 - <<'PY'
import pathlib
p = pathlib.Path("src/r2egym/agenthub/utils/utils.py")
if p.exists():
    t = p.read_text()
    if "optional HF Hub upload" not in t:
        t = t.replace(
            "from huggingface_hub import create_repo, upload_folder, HfFolder",
            "try:\n    from huggingface_hub import create_repo, upload_folder, HfFolder\nexcept ImportError:  # optional HF Hub upload, not local eval\n    create_repo = upload_folder = HfFolder = None")
        t = t.replace(
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "try:\n    from transformers import AutoModelForCausalLM, AutoTokenizer\nexcept ImportError:  # optional local model loading, not local eval\n    AutoModelForCausalLM = AutoTokenizer = None")
        p.write_text(t); print("  utils guards applied")
    else:
        print("  utils guards already present")
else:
    print("  WARN: utils.py not found")
PY

  # 2) force huggingface_hub<1.0 (HfFolder removed in >=1.0)
  uv pip install -p "$PY" 'huggingface_hub<1.0' || uv pip install -p "$PY" --no-deps 'huggingface_hub==0.27.1'
  # 3) swebench with get_eval_type
  uv pip install -p "$PY" --no-deps 'swebench==3.0.2'
  # 4) native ext rebuilds against container glibc/openssl
  uv pip install -p "$PY" --reinstall cffi cryptography

  echo "===== re-verify after FIX ====="
  "$PY" -c "from huggingface_hub import HfFolder; import swebench; from swebench.harness.log_parsers import get_eval_type; from r2egym.agenthub.runtime.local import LocalRuntime; from r2egym.agenthub.run import run_local_evaluation; import r2egym, kubernetes; print('  R2E EVAL CHAIN OK (post-fix)')" 2>&1 | tail -5 \
    && echo "########## R2E READY ##########" || echo "########## R2E STILL BROKEN ##########"
fi
echo "########## done ##########"
