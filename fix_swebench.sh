set -x
SWE=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents
R2E="$SWE/swe_r2e_gym_setup/R2E-Gym"; cd "$R2E" || exit 1
export UV_INSTALL_DIR="$SWE/swe_r2e_gym_setup/uv"; export PATH="$UV_INSTALL_DIR/bin:$PATH"
PY=venv/bin/python
$PY -c "import swebench; print('swebench before:', swebench.__version__)"
uv pip install -p "$PY" --no-deps 'swebench==3.0.2'
echo "============ VERIFY ============"
$PY -c "import swebench; from swebench.harness.log_parsers import get_eval_type; print('swebench', swebench.__version__, 'get_eval_type OK')" || echo "FAIL: swebench"
$PY -c "from r2egym.agenthub.runtime.local import LocalRuntime; from r2egym.agenthub.run import run_local_evaluation; import r2egym, kubernetes; print('EVAL CHAIN OK')" && echo "============ R2E EVAL READY ============" || echo "============ STILL BROKEN ============"
