#!/bin/bash
# Reinstall R2E-Gym with MINIMAL pin relaxation: restore exact pins, relax ONLY orjson+pygments
# (the two genuine resolver conflicts). This keeps swebench==3.0.2 (has get_eval_type),
# transformers==4.57.3, etc. -- the versions R2E-Gym's eval code actually expects.
# Plus keep the utils.py import-guards. Writes to the Lustre venv; no --container-save.
set -x
SWE=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents
R2E="$SWE/swe_r2e_gym_setup/R2E-Gym"
cd "$R2E" || { echo "FATAL: R2E missing"; exit 1; }
export UV_INSTALL_DIR="$SWE/swe_r2e_gym_setup/uv"
export PATH="$UV_INSTALL_DIR/bin:$PATH"
PY=venv/bin/python

# Restore original exact pins, then relax ONLY orjson + pygments
git checkout -- pyproject.toml
sed -i 's/"orjson==[0-9.]*"/"orjson"/; s/"pygments==[0-9.]*"/"pygments"/' pyproject.toml
echo "=== orjson/pygments/swebench/transformers lines ==="; grep -E '"(orjson|pygments|swebench|transformers|datasets)' pyproject.toml

# Re-apply the import guards in utils.py (git checkout above only touched pyproject, but be safe)
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/r2egym/agenthub/utils/utils.py"); t = p.read_text()
if "optional HF Hub upload" not in t:
    t = t.replace(
        "from huggingface_hub import create_repo, upload_folder, HfFolder",
        "try:\n    from huggingface_hub import create_repo, upload_folder, HfFolder\nexcept ImportError:  # only used for optional HF Hub upload, not local eval\n    create_repo = upload_folder = HfFolder = None")
    t = t.replace(
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "try:\n    from transformers import AutoModelForCausalLM, AutoTokenizer\nexcept ImportError:  # only used for optional local model loading, not local eval\n    AutoModelForCausalLM = AutoTokenizer = None")
    p.write_text(t); print("utils guards re-applied")
else:
    print("utils guards present")
PY

uv pip install -p "$PY" -e . --no-cache
uv pip install -p "$PY" --reinstall cffi cryptography

echo "============ VERIFY ============"
$PY -c "import swebench; from swebench.harness.log_parsers import get_eval_type; print('swebench', swebench.__version__, 'get_eval_type OK')" || echo "FAIL: swebench get_eval_type"
$PY -c "from r2egym.agenthub.runtime.local import LocalRuntime; from r2egym.agenthub.run import run_local_evaluation; import r2egym, kubernetes; print('EVAL CHAIN OK')" && echo "============ R2E EVAL READY ============" || echo "============ STILL BROKEN ============"
