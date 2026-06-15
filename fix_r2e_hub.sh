#!/bin/bash
# The R2E-Gym eval venv has huggingface_hub>=1.0 (HfFolder removed), but R2E-Gym's eval
# code imports HfFolder -> ImportError -> every eval fails. Force hub<1.0 (HfFolder present).
set -x
SWE=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents
cd "$SWE/swe_r2e_gym_setup/R2E-Gym" || { echo "FATAL: R2E venv missing"; exit 1; }
export UV_INSTALL_DIR="$SWE/swe_r2e_gym_setup/uv"
export PATH="$UV_INSTALL_DIR/bin:$PATH"
PY=venv/bin/python

echo "=== current hub ==="; $PY -c "import huggingface_hub as h; print(h.__version__)"
# Try a clean resolve to hub<1.0; if uv refuses on conflicts, force just hub (no-deps).
uv pip install -p "$PY" 'huggingface_hub<1.0' || uv pip install -p "$PY" --no-deps 'huggingface_hub==0.27.1'

echo "============ VERIFY ============"
$PY -c "import huggingface_hub as h; from huggingface_hub import HfFolder, create_repo, upload_folder; import r2egym, kubernetes; print('hub', h.__version__, '| HfFolder OK | r2egym OK')" \
  && echo "============ R2E EVAL FIXED ============" || echo "============ STILL BROKEN ============"
