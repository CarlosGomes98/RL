#!/bin/bash
# Rebuild ONLY the R2E-Gym eval harness. nv-R2E-Gym's pyproject exact-pins (==) conflict
# with its own deps' newer requirements (orjson, pygments, ...). Let r2e_gym.sh do clone/
# checkout/venv (it fails at the editable install), then relax the == pins to >= and install
# ourselves into the venv it created. Writes to the Lustre symlink target; no --container-save.
set -x
SWE=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents
cd "$SWE" || { echo "FATAL: no $SWE"; exit 1; }
S="$SWE/swe_r2e_gym_setup"

# Run the stock setup: installs uv/python, clones+checks out R2E-Gym, creates venv,
# then FAILS at `uv pip install -e .` (expected). Tolerate that failure.
SETUP_DIR="$S" UV_DIR="$S/uv" PYTHON_DIR="$S/python" R2E_GYM_DIR="$S/R2E-Gym" \
EVAL_HARNESS_REPO=https://github.com/sdevare-nv/nv-R2E-Gym.git EVAL_HARNESS_COMMIT=local-eval \
bash setup_scripts/r2e_gym.sh || echo "(stock r2e_gym.sh failed at install, as expected -- fixing pins)"

# Now relax pins and install ourselves into the venv r2e_gym.sh created.
export UV_INSTALL_DIR="$S/uv" UV_PYTHON_INSTALL_DIR="$S/python"
export PATH="$S/uv/bin:$PATH"
cd "$S/R2E-Gym" || { echo "FATAL: R2E-Gym clone missing"; exit 1; }

cp -f pyproject.toml pyproject.toml.bak
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("pyproject.toml")
# relax exact pins: "<pkg>[extras]==x.y.z"  ->  "<pkg>[extras]>=x.y.z"
p.write_text(re.sub(r'("[A-Za-z0-9_.\[\],-]+)==', r'\1>=', p.read_text()))
print("relaxed pyproject written")
PY
echo "=== relaxed deps (sample) ==="; grep -E '>=' pyproject.toml | head -20

uv pip install -p venv/bin/python -e . --no-cache
uv pip install -p venv/bin/python --reinstall cffi cryptography

echo "============ VERIFY ============"
venv/bin/python -c "import r2egym, kubernetes, orjson; print('r2e OK; orjson', orjson.__version__)" && echo "============ R2E OK ============" || echo "============ R2E STILL FAILING ============"
