#!/usr/bin/env bash
# One-shot backend setup for the single-container tutorial. Started from the
# notebook (Environment setup, step 4). Everything runs under the system
# python3 that already carries vLLM / PyTorch / ROCm in the official image.
#
# CONFIGURATION IS DRIVEN FROM THE ENVIRONMENT. The tutorial notebook sets the
# knobs below (model, service ports, GPU memory fraction, OTEL pins) and the
# Python packages in a readable "configure the backend" cell, then runs this
# script. Every value here falls back to a sensible default, so the script also
# runs standalone: `bash utils/setup_backend.sh`.
#
# FAILURE POLICY. Every step is checked. The first failure stops the script at
# that point, prints what failed (with the tail of that step's log), stops the
# services this run had started, and exits non-zero so the notebook cell fails.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Service logs go in LOG_DIR; the output of each setup step goes in LOG_DIR/steps.
LOG_DIR="${SETUP_LOG_DIR:-/root/logs}"
STEP_LOG_DIR="${LOG_DIR}/steps"
ORIG_PATH="${PATH}"
STARTED=()   # pkill -f patterns of the services this run launched

log(){ echo "[$(date +%H:%M:%S)] $*"; }

show_tail(){ # file [lines]
  if [ -s "$1" ]; then
    log "---- last ${2:-25} lines of $1 ----"
    tail -n "${2:-25}" "$1" | sed 's/^/    /'
    log "----"
  fi
}

fail(){
  trap - ERR
  if [ "${BASH_SUBSHELL:-0}" -gt 0 ]; then echo "ERROR: $*" >&2; exit 1; fi
  log "============================================================================"
  log "SETUP FAILED: $*"
  if [ "${#STARTED[@]}" -gt 0 ]; then
    local p
    for p in "${STARTED[@]}"; do pkill -f "$p" 2>/dev/null || true; done
    log "Stopped the services this run had started, so nothing is left half-up."
  fi
  log "Fix the issue above, then re-run this script. Logs: ${LOG_DIR}"
  log "============================================================================"
  exit 1
}

# Catch-all for commands that have no explicit check. Inside $(...) it only
# exits: a banner printed there would be captured into the variable, and the
# caller decides how to handle the failure.
on_err(){
  if [ "${BASH_SUBSHELL:-0}" -gt 0 ]; then exit "$1"; fi
  fail "unexpected error (exit $1) at line $2: $3"
}
trap 'on_err $? $LINENO "$BASH_COMMAND"' ERR
trap 'fail "interrupted"' INT TERM

run(){ # "description" command [args...]: run quietly, stop with its log on failure
  local desc="$1"; shift
  local logf rc=0
  logf="${STEP_LOG_DIR}/$(printf '%s' "$desc" | tr -cs 'A-Za-z0-9' '_' | cut -c1-60).log"
  "$@" >"$logf" 2>&1 </dev/null || rc=$?
  if [ "$rc" -ne 0 ]; then
    show_tail "$logf"
    fail "${desc} failed (exit ${rc}). Full log: ${logf}"
  fi
}

require_cmd(){ command -v "$1" >/dev/null 2>&1 || fail "required command '$1' was not found${2:+ ($2)}."; }

stop_existing(){ # pattern name: stop a previous instance and wait until it is gone
  pgrep -f "$1" >/dev/null 2>&1 || return 0
  log "  stopping the existing ${2}..."
  pkill -f "$1" 2>/dev/null || true
  local i
  for i in $(seq 1 30); do
    pgrep -f "$1" >/dev/null 2>&1 || return 0
    sleep 1
  done
  pkill -9 -f "$1" 2>/dev/null || true
  sleep 2
  if pgrep -f "$1" >/dev/null 2>&1; then fail "could not stop the existing ${2} (processes matching '$1')."; fi
}

port_free(){
  python3 - "$1" <<'PY'
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    sys.exit(1)
PY
}

ensure_port_free(){ # port name
  local i who
  for i in $(seq 1 10); do
    if port_free "$1"; then return 0; fi
    sleep 1
  done
  who="$(lsof -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | awk 'NR>1{print $1" (pid "$2")"}' | sort -u | tr '\n' ' ')" || who=""
  fail "port $1 needed by $2 is already in use${who:+ by: ${who}}. Stop that process or choose another port, then re-run."
}

diagnose(){ # logfile: one-line hint for well-known failure causes
  local f="${1:-}"
  if [ -z "$f" ] || [ ! -f "$f" ]; then return 0; fi
  if grep -qiE 'No space left on device' "$f"; then
    echo " Cause: the disk filled up (free space at ${HF_HUB_DIR})."
  elif grep -qiE 'out of memory|OutOfMemoryError' "$f"; then
    echo " Cause: GPU out of memory (another process holds the GPU, or lower GPU_MEMORY_UTILIZATION)."
  elif grep -qiE 'GatedRepoError|401 Client Error|403 Client Error|Access to model .* is restricted' "$f"; then
    echo " Cause: no access to ${HERMES_MODEL} on Hugging Face (set HF_TOKEN / accept the model license)."
  elif grep -qiE 'Address already in use' "$f"; then
    echo " Cause: the port is already in use."
  fi
  return 0
}

wait_http(){ # url name timeout [pid] [logfile]: ready when curl -sf succeeds
  local url="$1" name="$2" t="$3" pid="${4:-}" logf="${5:-}" w=0
  while [ "$w" -lt "$t" ]; do
    if curl -sf -o /dev/null --max-time 5 "$url"; then
      log "  ${name} is up."
      return 0
    fi
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      if [ -n "$logf" ]; then show_tail "$logf" 40; fi
      fail "${name} exited before becoming ready${logf:+ (log: ${logf})}.$(diagnose "$logf")"
    fi
    sleep 5
    w=$((w + 5))
    if [ $((w % 60)) -eq 0 ]; then log "  waiting for ${name}... ${w}s"; fi
  done
  if [ -n "$logf" ]; then show_tail "$logf" 40; fi
  fail "${name} did not become ready within ${t}s (${url})${logf:+; log: ${logf}}.$(diagnose "$logf")"
}

assert_alive(){ # pid name logfile: catch an early crash between steps
  if ! kill -0 "$1" 2>/dev/null; then
    show_tail "$3" 40
    fail "$2 exited unexpectedly (log: $3).$(diagnose "$3")"
  fi
}

HERMES_MODEL="${HERMES_MODEL:-meta-models/Muse-Glimmer-30B}"
VLLM_PORT="${VLLM_PORT:-8001}"
MLFLOW_PORT="${MLFLOW_PORT:-5004}"
PROM_PORT="${PROM_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-3600}"
VLLM_MIN_FREE_GB="${VLLM_MIN_FREE_GB:-80}"
VLLM_MIN_FREE_GB_CACHED="${VLLM_MIN_FREE_GB_CACHED:-10}"
HERMES_HOME_DIR="${HERMES_HOME:-${HOME}/.hermes}"
HERMES_INSTALL_URL="https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh"
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore DEBIAN_FRONTEND=noninteractive

# OpenTelemetry pins. The official vLLM image ships a coherent opentelemetry
# 1.40.0 stack, and lmcache (installed in the image) caps opentelemetry-api at
# <=1.40.0. Pin the whole otel stack to 1.40.0 so pip's resolver stays
# consistent; bumping only a couple of otel packages leaves the rest at 1.40.0
# and breaks pip check. The notebook exports OTEL_PINS; this default keeps the
# script self-contained when it is run on its own.
OTEL_PINS="${OTEL_PINS:-opentelemetry-api==1.40.0 opentelemetry-sdk==1.40.0 opentelemetry-proto==1.40.0 opentelemetry-semantic-conventions==0.61b0 opentelemetry-exporter-otlp-proto-common==1.40.0 opentelemetry-exporter-otlp-proto-grpc==1.40.0 opentelemetry-exporter-otlp-proto-http==1.40.0 opentelemetry-exporter-otlp==1.40.0}"

# =============================================================================
# Preflight: checks that need nothing running, so a doomed run stops in seconds.
# =============================================================================
log "Preflight checks..."
[ "$(id -u)" -eq 0 ] || fail "run this script as root (it installs OS packages and writes under /root)."
mkdir -p "${STEP_LOG_DIR}" || fail "cannot create the log directory ${STEP_LOG_DIR}."
[ "$(uname -m)" = "x86_64" ] || fail "this script downloads linux-amd64 builds, but the machine is $(uname -m)."
for c in python3 curl git tar df awk sed dpkg; do require_cmd "$c"; done
python3 -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("vllm") else 1)' \
  || fail "python3 cannot import vllm. Run this inside the official vLLM ROCm container."
[ -e /dev/kfd ] || fail "/dev/kfd is missing: the container cannot see the AMD GPU (start it with --device=/dev/kfd --device=/dev/dri)."
[ -f "${SCRIPT_DIR}/hermes_profiler.py" ] || fail "the dashboard app ${SCRIPT_DIR}/hermes_profiler.py is missing (re-run the notebook cell that fetches the support files)."

# --- Model weights: cache, disk space, Hugging Face access ------------------
# Measure the same cache vLLM downloads into (HF_HUB_CACHE / HF_HOME rules).
HF_HUB_DIR="$(python3 -c 'from huggingface_hub import constants; print(constants.HF_HUB_CACHE)' 2>/dev/null)" || HF_HUB_DIR=""
if [ -z "${HF_HUB_DIR}" ]; then HF_HUB_DIR="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"; fi
mkdir -p "${HF_HUB_DIR}" || fail "cannot create the Hugging Face cache directory ${HF_HUB_DIR}."
[ -w "${HF_HUB_DIR}" ] || fail "the Hugging Face cache directory ${HF_HUB_DIR} is not writable."

MODEL_CACHE="${HF_HUB_DIR}/models--${HERMES_MODEL//\//--}"
MODEL_STATE="$(python3 - "${MODEL_CACHE}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); snaps = root / "snapshots"
if not snaps.is_dir():
    print("absent"); sys.exit()
if any((root / "blobs").glob("*.incomplete")):
    print("partial"); sys.exit()
for snap in snaps.iterdir():
    index = snap / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError):
            continue
        if shards and all((snap / s).exists() for s in shards):
            print("complete"); sys.exit()
    elif (snap / "model.safetensors").exists():
        print("complete"); sys.exit()
print("partial" if any(snaps.rglob("*.safetensors")) else "absent")
PY
)" || fail "could not inspect the model cache at ${MODEL_CACHE}."

FREE_GB="$(df -PBG "${HF_HUB_DIR}" | awk 'NR==2{gsub(/G/,"",$4); print $4}')" || FREE_GB=""
[[ "${FREE_GB}" =~ ^[0-9]+$ ]] || fail "could not determine the free disk space at ${HF_HUB_DIR}."
if [ "${MODEL_STATE}" = "complete" ]; then NEED_GB="${VLLM_MIN_FREE_GB_CACHED}"; else NEED_GB="${VLLM_MIN_FREE_GB}"; fi
log "Model ${HERMES_MODEL}: ${MODEL_STATE} in ${HF_HUB_DIR}; ${FREE_GB} GB free (need >= ${NEED_GB} GB)."
if [ "${FREE_GB}" -lt "${NEED_GB}" ]; then
  fail "only ${FREE_GB} GB free at ${HF_HUB_DIR}; need >= ${NEED_GB} GB for ${HERMES_MODEL}. Free up space or point HF_HOME at a larger volume."
fi

if [ "${MODEL_STATE}" != "complete" ]; then
  if [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
    fail "HF_HUB_OFFLINE=1 but ${HERMES_MODEL} is not fully cached in ${HF_HUB_DIR}."
  fi
  HF_AUTH=()
  if [ -n "${HF_TOKEN:-}" ]; then HF_AUTH=(-H "Authorization: Bearer ${HF_TOKEN}"); fi
  HF_CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "${HF_AUTH[@]}" "https://huggingface.co/api/models/${HERMES_MODEL}")" || HF_CODE="000"
  case "${HF_CODE}" in
    200) ;;
    401|403)
      if [ -n "${HF_TOKEN:-}" ]; then
        fail "HF_TOKEN has no access to ${HERMES_MODEL} (HTTP ${HF_CODE}). Accept the license on its Hugging Face page or use a token that has access."
      fi
      fail "${HERMES_MODEL} is gated or does not exist (HTTP ${HF_CODE}). Export HF_TOKEN if it is gated, or check HERMES_MODEL." ;;
    404) fail "${HERMES_MODEL} was not found on Hugging Face (HTTP 404). Check HERMES_MODEL." ;;
    000) fail "cannot reach huggingface.co to download ${HERMES_MODEL} (no network or proxy?), and it is not cached." ;;
    *)   fail "unexpected HTTP ${HF_CODE} from the Hugging Face API for ${HERMES_MODEL}." ;;
  esac
fi

# =============================================================================
# OS and Python packages
# =============================================================================
APT_PKGS=(espeak-ng libsndfile1 procps lsof ripgrep)
MISSING_PKGS=()
for p in "${APT_PKGS[@]}"; do
  if ! dpkg -s "$p" >/dev/null 2>&1; then MISSING_PKGS+=("$p"); fi
done
if [ "${#MISSING_PKGS[@]}" -gt 0 ]; then
  require_cmd apt-get "needed to install ${MISSING_PKGS[*]}"
  log "Installing OS packages: ${MISSING_PKGS[*]}..."
  run "apt-get update" apt-get update -qq
  run "apt-get install" apt-get install -y -qq "${MISSING_PKGS[@]}"
fi
require_cmd pkill "from procps"
require_cmd pgrep "from procps"

# The notebook's configuration cell installs these before running this script;
# install them here too when they are missing, so a standalone run still works.
PY_MODULES="streamlit mlflow kokoro soundfile fastapi uvicorn plotly pandas ipywidgets matplotlib kaleido psutil requests opentelemetry.sdk"
missing_modules(){
  python3 - ${PY_MODULES} <<'PY'
import importlib.util, sys
missing = []
for name in sys.argv[1:]:
    try:
        found = importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        found = False
    if not found:
        missing.append(name)
print(" ".join(missing))
PY
}
MISSING_MODS="$(missing_modules)" || fail "could not check the installed Python packages."
if [ -n "${MISSING_MODS}" ]; then
  log "Installing Python packages (missing: ${MISSING_MODS})..."
  # shellcheck disable=SC2086
  run "pip install tutorial packages" python3 -m pip install -q \
    kokoro soundfile fastapi "uvicorn[standard]" \
    "streamlit>=1.30" "mlflow>=3.0.0" plotly pandas ipywidgets matplotlib kaleido \
    ${OTEL_PINS} psutil requests
  MISSING_MODS="$(missing_modules)" || fail "could not check the installed Python packages."
  [ -z "${MISSING_MODS}" ] || fail "pip finished but these modules still do not import: ${MISSING_MODS}."
fi

# =============================================================================
# Metrics: upstream Prometheus (native OTLP receiver) + Grafana
# =============================================================================
latest_release(){ # owner/repo -> version without the leading v
  local v
  v="$(curl -fsSL --max-time 30 "https://api.github.com/repos/$1/releases/latest" \
       | python3 -c 'import sys, json; print(json.load(sys.stdin)["tag_name"].lstrip("v"))')" || return 1
  [[ "$v" =~ ^[0-9]+\.[0-9]+ ]] || return 1
  printf '%s' "$v"
}

if [ ! -x /root/prom/prometheus ]; then
  if [ -z "${PROM_VER:-}" ]; then
    PROM_VER="$(latest_release prometheus/prometheus)" \
      || fail "could not resolve the latest Prometheus release from the GitHub API (network down or rate-limited). Set PROM_VER=<version> to skip the lookup."
  fi
  log "Installing Prometheus ${PROM_VER}..."
  TMP_DL="$(mktemp -d)"
  run "Download Prometheus ${PROM_VER}" curl -fSL --retry 3 -o "${TMP_DL}/prom.tgz" \
    "https://github.com/prometheus/prometheus/releases/download/v${PROM_VER}/prometheus-${PROM_VER}.linux-amd64.tar.gz"
  run "Extract Prometheus" tar xzf "${TMP_DL}/prom.tgz" -C "${TMP_DL}"
  [ -x "${TMP_DL}/prometheus-${PROM_VER}.linux-amd64/prometheus" ] || fail "the Prometheus archive did not contain the expected binary."
  rm -rf /root/prom && mv "${TMP_DL}/prometheus-${PROM_VER}.linux-amd64" /root/prom
  rm -rf "${TMP_DL}"
fi
cat > /root/prom/prometheus.yml <<'CFG'
global:
  scrape_interval: 15s
otlp:
  promote_resource_attributes: [service.instance.id, service.name]
storage:
  tsdb:
    out_of_order_time_window: 30m
CFG
stop_existing '/root/prom/prometheus' "Prometheus"
ensure_port_free "${PROM_PORT}" "Prometheus"
log "Starting Prometheus on :${PROM_PORT}..."
nohup /root/prom/prometheus --config.file=/root/prom/prometheus.yml \
  --web.listen-address="0.0.0.0:${PROM_PORT}" --web.enable-otlp-receiver \
  --storage.tsdb.path=/root/prom/data > "${LOG_DIR}/prometheus.log" 2>&1 &
PROM_PID=$!
STARTED+=('/root/prom/prometheus')
wait_http "http://localhost:${PROM_PORT}/-/ready" "Prometheus" 120 "${PROM_PID}" "${LOG_DIR}/prometheus.log"

if [ ! -x /root/grafana/bin/grafana ]; then
  if [ -z "${GRAF_VER:-}" ]; then
    GRAF_VER="$(latest_release grafana/grafana)" \
      || fail "could not resolve the latest Grafana release from the GitHub API (network down or rate-limited). Set GRAF_VER=<version> to skip the lookup."
  fi
  log "Installing Grafana ${GRAF_VER}..."
  TMP_DL="$(mktemp -d)"
  run "Download Grafana ${GRAF_VER}" curl -fSL --retry 3 -o "${TMP_DL}/graf.tgz" \
    "https://dl.grafana.com/oss/release/grafana-${GRAF_VER}.linux-amd64.tar.gz"
  run "Extract Grafana" tar xzf "${TMP_DL}/graf.tgz" -C "${TMP_DL}"
  GRAF_DIR="$(find "${TMP_DL}" -mindepth 1 -maxdepth 1 -type d -name 'grafana*' -print -quit)"
  [ -n "${GRAF_DIR}" ] && [ -x "${GRAF_DIR}/bin/grafana" ] || fail "the Grafana archive did not contain the expected bin/grafana."
  rm -rf /root/grafana && mv "${GRAF_DIR}" /root/grafana
  rm -rf "${TMP_DL}"
fi
mkdir -p /root/grafana/conf/provisioning/datasources
cat > /root/grafana/conf/provisioning/datasources/prom.yaml <<DS
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:${PROM_PORT}
    isDefault: true
DS
cat > /root/grafana.ini <<INI
[server]
http_addr = 0.0.0.0
http_port = ${GRAFANA_PORT}
[auth.anonymous]
enabled = true
[security]
admin_user = admin
admin_password = admin
INI
stop_existing 'grafana server' "Grafana"
ensure_port_free "${GRAFANA_PORT}" "Grafana"
log "Starting Grafana on :${GRAFANA_PORT}..."
( cd /root/grafana && exec nohup ./bin/grafana server --config=/root/grafana.ini \
    --homepath=/root/grafana ) > "${LOG_DIR}/grafana.log" 2>&1 &
GRAF_PID=$!
STARTED+=('grafana server')
wait_http "http://localhost:${GRAFANA_PORT}/api/health" "Grafana" 120 "${GRAF_PID}" "${LOG_DIR}/grafana.log"

# =============================================================================
# vLLM: the agent's model, served with AITER on the MI300X. It starts here so
# the weight download overlaps the rest of the setup. Later steps check that it
# is still running, and the script waits for it to be ready further down.
# =============================================================================
stop_existing 'vllm.entrypoints.openai.api_server' "vLLM server"
ensure_port_free "${VLLM_PORT}" "vLLM"
log "Starting vLLM (${HERMES_MODEL})$([ "${MODEL_STATE}" = complete ] || echo '; the first start downloads the weights')..."
HIP_VISIBLE_DEVICES=0 VLLM_ROCM_USE_AITER=1 \
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "${HERMES_MODEL}" --served-model-name "${HERMES_MODEL}" \
  --tensor-parallel-size 1 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer --attention-backend ROCM_AITER_FA \
  --generation-config auto --enable-prefix-caching \
  --host 0.0.0.0 --port "${VLLM_PORT}" > "${LOG_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!
STARTED+=('vllm.entrypoints.openai.api_server')

# =============================================================================
# MLflow: stores the execution traces
# =============================================================================
stop_existing 'mlflow server' "MLflow"
ensure_port_free "${MLFLOW_PORT}" "MLflow"
log "Starting MLflow on :${MLFLOW_PORT}..."
nohup python3 -m mlflow server --host 0.0.0.0 --port "${MLFLOW_PORT}" \
  --backend-store-uri sqlite:////root/mlflow.db --allowed-hosts '*' > "${LOG_DIR}/mlflow.log" 2>&1 &
MLFLOW_PID=$!
STARTED+=('mlflow server')
wait_http "http://localhost:${MLFLOW_PORT}/health" "MLflow" 180 "${MLFLOW_PID}" "${LOG_DIR}/mlflow.log"
assert_alive "${VLLM_PID}" "vLLM" "${LOG_DIR}/vllm.log"

# =============================================================================
# Hermes agent
# =============================================================================
# The installer publishes the launcher in ~/.local/bin and only adds that to
# PATH through shell rc files, which this non-interactive shell never reads.
add_hermes_to_path(){
  local d
  for d in "${HOME}/.local/bin" "${HERMES_HOME_DIR}/hermes-agent/.hermes/bin"; do
    if [ -x "${d}/hermes" ]; then
      case ":${PATH}:" in *":${d}:"*) ;; *) PATH="${d}:${PATH}" ;; esac
      export PATH
      return 0
    fi
  done
  return 1
}
install_hermes(){
  log "Running the Hermes installer..."
  run "Install the Hermes agent" bash -o pipefail -c "curl -fsSL '${HERMES_INSTALL_URL}' | bash -s -- --skip-setup"
}

if ! command -v hermes >/dev/null 2>&1; then add_hermes_to_path || true; fi
if ! command -v hermes >/dev/null 2>&1; then
  install_hermes
  add_hermes_to_path || true
  command -v hermes >/dev/null 2>&1 \
    || fail "the Hermes installer finished but no 'hermes' command exists (checked PATH, ~/.local/bin, ${HERMES_HOME_DIR}/hermes-agent/.hermes/bin)."
fi
run "Check the Hermes CLI runs" hermes --version
log "Hermes CLI: $(command -v hermes)"

# Later notebook cells run `hermes` with the kernel's PATH, which may not
# include ~/.local/bin either, so link the launcher into /usr/local/bin.
HERMES_BIN="$(command -v hermes)"
if ! PATH="${ORIG_PATH}" command -v hermes >/dev/null 2>&1; then
  ln -sf "${HERMES_BIN}" /usr/local/bin/hermes || fail "could not link ${HERMES_BIN} into /usr/local/bin."
  log "  linked hermes into /usr/local/bin so later notebook cells find it."
fi

# Edge TTS is Hermes' default text-to-speech provider, but its package is not
# part of the default install. Hermes installs it on first use, and only
# processes started after that can load it, so the first TTS request would fail
# (see NousResearch/hermes-agent#122097). Install it up front instead. This
# rebuilds Hermes' Python environment, so it must run before that environment
# is located and the telemetry packages are installed into it.
log "Installing Edge TTS support for Hermes..."
run "Install the Hermes edge-tts extra" hermes pm install --extra edge-tts

# Resolve the interpreter Hermes actually runs its code with: the plugin's
# dependencies and amdsmi must be importable there, or the agent emits no
# telemetry and the dashboard stays empty with no error. Current installers
# record that environment in installs/<key>/facts.json; older ones exec a
# venv/bin/python (or .venv) straight from the launcher.
find_hermes_python(){
  HERMES_HOME_DIR="${HERMES_HOME_DIR}" python3 - "${HERMES_BIN}" <<'PY'
import hashlib, json, os, re, sys
from pathlib import Path

home = Path(os.environ["HERMES_HOME_DIR"]).expanduser()
text = ""
try:
    text = Path(sys.argv[1]).resolve().read_text(errors="replace")
except OSError:
    pass

def recorded_python(facts):
    try:
        env = json.loads(facts.read_text(encoding="utf-8-sig"))["packages"]["venv"]["environment"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return Path(env) / "bin" / "python" if isinstance(env, str) else None

candidates = []
roots = re.findall(r"""sys\.path\.insert\(0,\s*['"]*(/[^'"]+)""", text) + [str(home / "hermes-agent")]
for root in roots:
    key = hashlib.sha256(str(Path(root).resolve()).encode("utf-8")).hexdigest()[:16]
    candidates.append(recorded_python(home / "installs" / key / "facts.json"))
facts_files = sorted(home.glob("installs/*/facts.json"), key=lambda p: p.stat().st_mtime, reverse=True)
candidates += [recorded_python(f) for f in facts_files]
candidates += [Path(m) for m in re.findall(r"""(/[^\s'"]+/\.?venv/bin/python[0-9.]*)""", text)]
for base in (home / "hermes-agent", Path("/usr/local/lib/hermes-agent")):
    candidates += [base / "venv" / "bin" / "python", base / ".venv" / "bin" / "python"]

for c in candidates:
    if c is not None and os.access(c, os.X_OK):
        print(c)
        sys.exit(0)
sys.exit(1)
PY
}

HVPY="$(find_hermes_python)" || HVPY=""
if [ -z "${HVPY}" ]; then
  # A missing environment means the install is incomplete; running the
  # installer again repairs it and creates the environment.
  log "Hermes' Python environment was not found; re-running the installer to create it..."
  install_hermes
  HVPY="$(find_hermes_python)" || HVPY=""
fi
[ -n "${HVPY}" ] || fail "Hermes' Python environment still does not exist after reinstalling (looked in ${HERMES_HOME_DIR}/installs/*/facts.json and the venv/.venv layouts)."
run "Check Hermes' Python environment" "${HVPY}" -c 'import sys'
log "Hermes venv python: ${HVPY}"
run "Check Edge TTS in Hermes' environment" "${HVPY}" -c 'import edge_tts'

# The environment is created by uv and has no pip of its own.
if ! "${HVPY}" -m pip --version >/dev/null 2>&1; then
  log "  adding pip to Hermes' environment..."
  if ! "${HVPY}" -m ensurepip --upgrade >"${STEP_LOG_DIR}/ensurepip.log" 2>&1; then
    GET_PIP="$(mktemp --suffix=.py)"
    run "Download get-pip.py" curl -fsSL --retry 3 -o "${GET_PIP}" https://bootstrap.pypa.io/get-pip.py
    run "Bootstrap pip into Hermes' environment" "${HVPY}" "${GET_PIP}"
    rm -f "${GET_PIP}"
  fi
  run "Check pip in Hermes' environment" "${HVPY}" -m pip --version
fi
assert_alive "${VLLM_PID}" "vLLM" "${LOG_DIR}/vllm.log"

# --- hermes-otel plugin (latest release tag) --------------------------------
PLUGIN_DIR="${HERMES_HOME_DIR}/plugins/hermes_otel"
rm -rf "${PLUGIN_DIR}"
mkdir -p "$(dirname "${PLUGIN_DIR}")"
run "Clone hermes-otel" git clone -q https://github.com/briancaffey/hermes-otel.git "${PLUGIN_DIR}"
HOTEL="$(git -C "${PLUGIN_DIR}" tag -l 'hermes-otel-v*' | sort -V | tail -1)"
if [ -n "${HOTEL}" ]; then
  run "Check out hermes-otel ${HOTEL}" git -C "${PLUGIN_DIR}" checkout -q "${HOTEL}"
else
  log "  no hermes-otel release tag found; using the default branch."
fi
log "hermes-otel ${HOTEL:-default-branch}"
run "Install hermes-otel for python3" python3 -m pip install -q -e "${PLUGIN_DIR}"
# shellcheck disable=SC2086
run "Install hermes-otel into Hermes' environment" "${HVPY}" -m pip install -q -e "${PLUGIN_DIR}" psutil requests ${OTEL_PINS}
run "Check telemetry packages in Hermes' environment" "${HVPY}" -c 'import opentelemetry.sdk, psutil, requests'

# amdsmi (the GPU metrics source) ships with ROCm in the system Python, not in
# Hermes' environment, so add its location to that environment's path.
AMDSMI_PARENT="$(python3 -c 'import amdsmi, os; print(os.path.dirname(os.path.dirname(amdsmi.__file__)))' 2>/dev/null)" \
  || fail "python3 cannot import amdsmi, which the GPU metrics need (it ships with ROCm in the official image)."
HV_SITE="$("${HVPY}" -c 'import site; print(site.getsitepackages()[0])')" \
  || fail "could not find the site-packages directory of ${HVPY}."
echo "${AMDSMI_PARENT}" > "${HV_SITE}/amdsmi_system.pth"
run "Check amdsmi in Hermes' environment" "${HVPY}" -c 'import amdsmi'

# Point the plugin's metrics at Prometheus (OTLP) and its traces at MLflow.
# hermes-otel rewrites the /v1/traces suffix per signal, so aiming a metrics
# backend at Prometheus's /api/v1/otlp/v1/traces makes metrics land on
# /api/v1/otlp/v1/metrics, which is the Prometheus OTLP metrics endpoint.
cat > "${PLUGIN_DIR}/config.yaml" <<YML
host_metrics: true
host_metrics_gpu: amd
host_metrics_interval_ms: 100
flush_interval_ms: 3000
backends:
  - type: otlp
    name: mlflow-traces
    endpoint: http://127.0.0.1:${MLFLOW_PORT}/v1/traces
    metrics: false
    # MLflow 3.x OTLP trace ingestion requires the target experiment id as an
    # HTTP header; without it every span batch is rejected with HTTP 422 and no
    # traces are stored (the profiling dashboard would stay empty). "0" is the
    # Default experiment MLflow creates on first start.
    headers:
      x-mlflow-experiment-id: "0"
  - type: lgtm
    name: prometheus-metrics
    endpoint: http://127.0.0.1:${PROM_PORT}/api/v1/otlp/v1/traces
    traces: false
    metrics: true
YML

# --- Hermes model configuration, verified after each write ------------------
hermes_set(){ # key value
  local got
  run "hermes config set $1" hermes config set "$1" "$2"
  got="$(hermes config get "$1" 2>&1)" || fail "'hermes config get $1' failed: ${got}"
  [[ "${got,,}" == *"${2,,}"* ]] || fail "'hermes config set $1 $2' did not take effect; 'hermes config get $1' returned: ${got}"
}
log "Configuring Hermes to use ${HERMES_MODEL} at http://localhost:${VLLM_PORT}/v1..."
hermes_set model.provider custom
hermes_set model.base_url "http://localhost:${VLLM_PORT}/v1"
hermes_set model.default "${HERMES_MODEL}"
hermes_set model.max_tokens 16384
hermes_set compression.enabled false
hermes_set display.show_reasoning false
# Hermes hides tools outside its built-in core set (such as the custom kokoro_tts
# tool) behind a tool_search / tool_call bridge, and rejects a direct call by
# name. Turn the bridge off so every tool, including kokoro_tts, can be called
# directly. Hermes reads false as "off".
hermes_set tools.tool_search.enabled false

# =============================================================================
# Wait for the model, then prove the agent can reach it
# =============================================================================
log "Waiting for vLLM to load the model (first run downloads weights)..."
wait_http "http://localhost:${VLLM_PORT}/health" "vLLM" "${VLLM_TIMEOUT}" "${VLLM_PID}" "${LOG_DIR}/vllm.log"
SERVED="$(curl -sf --max-time 10 "http://localhost:${VLLM_PORT}/v1/models")" || fail "vLLM is up but /v1/models did not respond."
[[ "${SERVED}" == *"\"${HERMES_MODEL}\""* ]] || fail "vLLM is up but does not serve '${HERMES_MODEL}'. /v1/models returned: ${SERVED:0:300}"

# Run one real agent turn through the configured provider, so a broken model
# setup (for example "Hermes is not connected to any AI provider") fails here
# rather than in the agent cells. It runs before the telemetry plugin is
# enabled, so it leaves no trace in MLflow.
log "Checking that the Hermes agent can reach the model..."
SMOKE_LOG="${STEP_LOG_DIR}/hermes_smoke_test.log"
SMOKE_RC=0
( cd /root && timeout 600 hermes chat --yolo --oneshot -q "Reply with the single word OK. Do not call any tools." ) \
  > "${SMOKE_LOG}" 2>&1 </dev/null || SMOKE_RC=$?
if [ "${SMOKE_RC}" -ne 0 ] || grep -qiE 'not connected to any AI provider|Traceback \(most recent call last\)|AuthError|Connection error' "${SMOKE_LOG}"; then
  show_tail "${SMOKE_LOG}" 30
  fail "the Hermes agent could not complete a turn against ${HERMES_MODEL} (exit ${SMOKE_RC}; log: ${SMOKE_LOG})."
fi
log "  Hermes agent reached the model."

run "Enable the hermes_otel plugin" hermes plugins enable hermes_otel --allow-tool-override

# =============================================================================
# Profiling dashboard (Streamlit)
# =============================================================================
# The show_session_overview() cells render an inline overview, but the full
# five-tab dashboard is a Streamlit app. Launch it from utils/ so Streamlit
# resolves utils/.streamlit/config.toml (the AMD theme); from any other CWD the
# theme is silently dropped.
stop_existing 'streamlit run' "dashboard"
ensure_port_free "${DASHBOARD_PORT}" "the dashboard"
log "Launching the profiling dashboard on :${DASHBOARD_PORT}..."
( cd "${SCRIPT_DIR}" && exec nohup python3 -m streamlit run hermes_profiler.py \
    --server.address 0.0.0.0 --server.port "${DASHBOARD_PORT}" --server.headless true ) \
  > "${LOG_DIR}/dashboard.log" 2>&1 &
DASH_PID=$!
STARTED+=('streamlit run')
wait_http "http://localhost:${DASHBOARD_PORT}/_stcore/health" "Dashboard" 120 "${DASH_PID}" "${LOG_DIR}/dashboard.log"

log "Backend ready. Dashboard :${DASHBOARD_PORT}  vLLM :${VLLM_PORT}  MLflow :${MLFLOW_PORT}  Grafana :${GRAFANA_PORT}  Prometheus :${PROM_PORT}"
log "Logs: ${LOG_DIR}"
