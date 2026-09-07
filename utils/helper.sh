#!/bin/bash

# ===========================================================================
# Configuration
# ===========================================================================
HERMES_WORKSPACE_DIR="$HOME/.hermes/workspace"
mkdir -p "$HERMES_WORKSPACE_DIR"
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# This script lives in utils/, so resolve its siblings (clear_cache.sh,
# kokoro_server.py, hermes_profiler.py, the profiling patch) relative to the
# script itself. WORKSPACE_DIR is the repo root, where the notebook, the env/
# venv, input_text.txt and outputs/ live, and where the agent's terminal.cwd
# points. Resolving paths this way lets the script be invoked from anywhere,
# e.g. `bash utils/helper.sh` from the repo root.
UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$UTILS_DIR/.." && pwd)"
cd "$WORKSPACE_DIR"

export PATH="$HOME/.local/bin:$PATH"

# Source ~/.bashrc and repair cache ownership, both guarded:
#   * ~/.bashrc short-circuits on non-interactive shells on Debian/Ubuntu, and
#     `set -u` inside a user's rc file could terminate this script, so it is
#     sourced defensively and only when it exists.
#   * A root-run setup step can leave $HOME/.cache root-owned, causing the HF
#     download and the Playwright install to fail with EACCES. The chown is
#     skipped when the cache is already owned correctly, so the common path
#     costs nothing, and it is only attempted where sudo is available.
if [ -f "$HOME/.bashrc" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.bashrc" || true
fi

mkdir -p "$HOME/.cache"
if [ ! -O "$HOME/.cache" ] && command -v sudo >/dev/null 2>&1; then
    echo "[INFO] Repairing ownership of $HOME/.cache ..."
    sudo chown -R "$(id -un):$(id -gn)" "$HOME/.cache" || \
        echo "[WARN] Could not chown $HOME/.cache; continuing."
fi

export HF_HOME="$HOME/.cache/huggingface"
sudo chown -R $USER:$USER "$HOME/.cache/huggingface"
HERMES_GPU="0"   # Muse-Glimmer-30B runs on GPU 0

# vLLM image: the official ROCm release image. Muse-Glimmer support ships in
# vLLM v0.28.0, so the release image is used directly with no local build or
# source overlay.
IMAGE_NAME="vllm/vllm-openai-rocm:v0.28.0"
VLLM_HERMES_PORT=8001

SYSTEM_IP=$(ip route get 1 2>/dev/null | awk '{print $7; exit}' || ip route get 8.8.8.8 | awk '{print $7; exit}')

# Clear caches before starting.
bash "$UTILS_DIR/clear_cache.sh"

HERMES_MODEL="meta-models/Muse-Glimmer-30B"

# Local Kokoro TTS server (sequential + batched inference modes).
KOKORO_PORT=8092
KOKORO_ENV="$WORKSPACE_DIR/env"
KOKORO_SERVER="$UTILS_DIR/kokoro_server.py"

# ===========================================================================
# Helpers and lifecycle management
# ===========================================================================

cleanup() {
    # Exit code: 0 from the trap (clean Ctrl+C / TERM), non-zero when called by fail().
    local exit_code="${1:-0}"
    echo -e "\n[INFO] Cleaning up containers and background services..."

    echo "[INFO] Stopping Streamlit dashboard..."
    if [ -n "$STREAMLIT_PID" ]; then
        kill "$STREAMLIT_PID" >/dev/null 2>&1
    fi
    sudo fuser -k 8501/tcp >/dev/null 2>&1

    echo "[INFO] Stopping MLflow server..."
    if [ -n "$MLFLOW_PID" ]; then
        kill "$MLFLOW_PID" >/dev/null 2>&1
    fi
    sudo fuser -k 5004/tcp >/dev/null 2>&1

    echo "[INFO] Stopping Kokoro TTS server..."
    if [ -n "$KOKORO_PID" ]; then
        kill "$KOKORO_PID" >/dev/null 2>&1
    fi
    sudo fuser -k ${KOKORO_PORT}/tcp >/dev/null 2>&1

    echo "[INFO] Stopping hermes_service container..."
    sudo docker stop hermes_service >/dev/null 2>&1
    sudo docker rm hermes_service >/dev/null 2>&1

    echo "[INFO] Removing profiling artifacts cache..."
    rm -rf "$HOME/profiling_cache" >/dev/null 2>&1
    echo "[INFO] Cleanup complete. Exiting."
    exit "$exit_code"
}

# Report a fatal setup failure, dump the offending log, tear everything down,
# and exit non-zero so the caller knows the run did not come up cleanly.
fail() {
    local service_name="$1"
    local log_file="$2"
    echo -e "\n[FATAL] $service_name failed to start properly. Aborting setup." >&2
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo "----- last 40 lines of $log_file -----" >&2
        tail -n 40 "$log_file" >&2
        echo "--------------------------------------" >&2
    fi
    cleanup 1
}

# Declared up front so cleanup can reference them safely even if Ctrl+C arrives
# before the corresponding server is started.
MLFLOW_PID=""
KOKORO_PID=""
STREAMLIT_PID=""

# Catch Ctrl+C and termination so containers are always cleaned up.
trap cleanup INT TERM

wait_for_vllm_readiness() {
    local port=$1
    local service_name=$2
    local timeout=600
    local counter=0

    echo "[INFO] Waiting for $service_name to load weights and start its API on port $port..."
    while true; do
        status_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/v1/models)
        status_code="${status_code:-000}"

        if [ "$status_code" -eq 200 ]; then
            echo "[OK] $service_name is active and responsive."
            return 0
        fi

        sleep 5
        counter=$((counter + 5))
        if [ $counter -ge $timeout ]; then
            echo "[ERROR] Timeout waiting for $service_name to respond."
            return 1
        fi
    done
}

# ===========================================================================
# vLLM engine
# ===========================================================================

# Remove any leftover containers from a previous run.
sudo docker rm -f hermes_service >/dev/null 2>&1

# Pull the official release image if it is not already present locally.
if [ -z "$(sudo docker images -q $IMAGE_NAME)" ]; then
    echo "[INFO] Image $IMAGE_NAME not found locally. Pulling..."
    sudo docker pull "$IMAGE_NAME"
else
    echo "[OK] Image $IMAGE_NAME found locally. Skipping pull."
fi

echo "[INFO] Launching hermes_service (vLLM)..."

sudo docker run -d \
    --ipc=host \
    --network=host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    --name hermes_service \
    -e HIP_VISIBLE_DEVICES=$HERMES_GPU \
    -e VLLM_ROCM_USE_AITER=1 \
    -v "$HERMES_WORKSPACE_DIR":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    --entrypoint /bin/bash \
    "$IMAGE_NAME" -c \
    "python3 -m vllm.entrypoints.openai.api_server \
        --model $HERMES_MODEL \
        --port $VLLM_HERMES_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.6 \
        --enable-auto-tool-choice \
        --tool-call-parser muse_glimmer \
        --reasoning-parser muse_glimmer \
        --attention-backend ROCM_AITER_FA \
        --generation-config auto \
        --enable-prefix-caching \
        --host 0.0.0.0"

echo "[INFO] Verifying container runtimes..."
if ! wait_for_vllm_readiness $VLLM_HERMES_PORT "hermes_service vLLM engine"; then
    echo "----- last 40 lines of hermes_service container logs -----" >&2
    sudo docker logs --tail 40 hermes_service >&2 2>&1
    echo "---------------------------------------------------------" >&2
    fail "hermes_service vLLM engine" ""
fi

# GPU numbers for the profiling patch come from amdsmi queried directly
# in-process (see hermes_otel's gpu_probe.py / host_metrics.py), not from an
# external exporter container. amdsmi is installed into the Hermes venv below.

# ===========================================================================
# Hermes toolchain and MLflow integration
# ===========================================================================
echo "[INFO] Installing MLflow and OpenTelemetry dependencies..."

# Bootstrap a usable Python toolchain before anything attempts pip install.
#
# Some AMD Dev Cloud ROCm images (e.g. rocm714-vllm-0.27.1-omni, Ubuntu 24.04,
# Python 3.12.3) ship with no pip and no ensurepip for the system interpreter
# and mark it PEP 668 externally-managed. In that state every `python3 -m pip`
# below fails with "No module named pip", which surfaces far downstream as a
# misleading "MLflow server failed to start". Bootstrapping here keeps the
# failure local and its message accurate.
ensure_python_toolchain() {
    local need_pip=0 need_venv=0
    python3 -m pip --version  >/dev/null 2>&1 || need_pip=1
    python3 -m venv --help    >/dev/null 2>&1 || need_venv=1

    if [ "$need_pip" -eq 0 ] && [ "$need_venv" -eq 0 ]; then
        echo "[OK] Python toolchain present ($(python3 -m pip --version 2>&1 | head -1))."
        return 0
    fi

    echo "[INFO] Bootstrapping Python toolchain (pip=$need_pip venv=$need_venv)..."
    # Run apt as root when we are not already root, and only if sudo exists.
    local as_root=""
    if [ "$(id -u)" -ne 0 ]; then
        command -v sudo >/dev/null 2>&1 && as_root="sudo"
    fi
    if command -v apt-get >/dev/null 2>&1; then
        $as_root apt-get update -qq >/dev/null 2>&1 || true
        DEBIAN_FRONTEND=noninteractive $as_root apt-get install -y -qq \
            python3-pip python3-venv >/dev/null 2>&1 || true
    fi

    # Fall back to the official bootstrap when the distro packages are absent.
    if ! python3 -m pip --version >/dev/null 2>&1; then
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
            && python3 /tmp/get-pip.py --break-system-packages >/dev/null 2>&1 || true
        rm -f /tmp/get-pip.py
    fi

    if python3 -m pip --version >/dev/null 2>&1; then
        echo "[OK] pip available: $(python3 -m pip --version 2>&1 | head -1)"
    else
        echo "[FATAL] Could not bootstrap pip for $(command -v python3)."
        echo "        Install python3-pip and python3-venv, then re-run this script."
        exit 1
    fi

    if ! python3 -m venv --help >/dev/null 2>&1; then
        echo "[FATAL] python3 venv module unavailable; install python3-venv and re-run."
        exit 1
    fi
}

ensure_python_toolchain

# PEP 668 marks the system interpreter externally-managed on Ubuntu 24.04, so a
# plain `pip install` is refused. These are ephemeral workshop hosts and the
# script already owns the system Python, so opt out explicitly.
PIP_SYS_FLAGS=""
if python3 -c "import sys,sysconfig,os; \
sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path('stdlib'), \
'EXTERNALLY-MANAGED')) else 1)" 2>/dev/null; then
    PIP_SYS_FLAGS="--break-system-packages"
    echo "[INFO] Interpreter is PEP 668 externally-managed; using --break-system-packages."
fi

# Distro-installed Python packages carry no RECORD file, so when pip needs to
# upgrade one to satisfy a dependency it cannot uninstall it and aborts the
# whole transaction:
#
#   ERROR: Cannot uninstall typing_extensions 4.10.0, RECORD file not found.
#          Hint: The package was installed by debian.
#
# mlflow 3.13.0 pulls a newer typing_extensions than the apt-shipped 4.10.0.
# --ignore-installed on just the offending names lets pip shadow them in
# site-packages without removing the apt copy. It is scoped deliberately: a
# blanket --ignore-installed would redownload the entire dependency tree.
PIP_SHADOW_DEBIAN="--ignore-installed typing_extensions"

# opentelemetry-exporter-otlp-proto-http is required: the hermes-otel plugin
# ships traces to MLflow over the OTLP/HTTP protobuf endpoint. Without it the
# plugin loads and prints its banner but exports nothing, so the dashboard sits
# empty with no error.
python3 -m pip install -q $PIP_SYS_FLAGS $PIP_SHADOW_DEBIAN \
  mlflow==3.13.0 opentelemetry-sdk==1.42.1 \
  opentelemetry-exporter-otlp-proto-http==1.42.1

# Verify the install here with an accurate message rather than letting it
# surface later as a confusing "MLflow server failed to start". Check the OTLP
# exporter too: without it the plugin exports nothing silently.
if ! python3 -c "import mlflow" 2>/dev/null; then
    echo "[FATAL] mlflow did not install into $(command -v python3)."
    echo "        Re-run without -q to see the error:"
    echo "        python3 -m pip install $PIP_SYS_FLAGS $PIP_SHADOW_DEBIAN mlflow==3.13.0"
    exit 1
fi
if ! python3 -c "from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter" 2>/dev/null; then
    echo "[FATAL] The OTLP/HTTP span exporter is not importable."
    echo "        Traces would be silently dropped, so stopping here."
    exit 1
fi
echo "[OK] MLflow $(python3 -c 'import mlflow; print(mlflow.__version__)') and the OTLP exporter are installed."

echo "[INFO] Launching MLflow server on port 5004..."
# --serve-artifacts / --artifacts-destination: without a configured artifact
# store MLflow records the RUN but cannot serve its artifacts, so the dashboard
# and the Traces view come up with missing detail. Keep --allowed-hosts "*",
# which the dashboard needs when it is reached over the server IP rather than
# localhost.
python3 -m mlflow server \
  --host 0.0.0.0 \
  --port 5004 \
  --backend-store-uri sqlite:///mlflow.db \
  --serve-artifacts \
  --artifacts-destination ./mlflow_artifacts \
  --allowed-hosts "*" > mlflow_server.log 2>&1 &

MLFLOW_PID=$!
echo "[INFO] MLflow server started (PID $MLFLOW_PID)."

echo "[INFO] Waiting for MLflow server /health on port 5004..."
mlflow_ready=0
for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:5004/health")
    code="${code:-000}"
    if [ "$code" -eq 200 ]; then
        echo "[OK] MLflow server is up."
        mlflow_ready=1
        break
    fi
    # Fail fast if the background process already died.
    if ! kill -0 "$MLFLOW_PID" 2>/dev/null; then
        break
    fi
    sleep 2
done
if [ "$mlflow_ready" -ne 1 ]; then
    fail "MLflow server" "$WORKSPACE_DIR/mlflow_server.log"
fi

# ===========================================================================
# Hermes Installation & Configuration
# ===========================================================================
sudo chown -R $(whoami):$(whoami) "$HOME/.hermes"
if ! command -v hermes &> /dev/null && [ ! -f "$HOME/.local/bin/hermes" ]; then
    echo "[INFO] Installing Hermes agent..."
    # The Hermes installer needs npm for its Node-based TUI. On a bare image
    # npm is absent and the install completes with a broken front end, so it is
    # provisioned first, conditionally: a machine that already has npm (and the
    # container image) does not pay for an apt round-trip, and a failure here
    # does not abort the whole setup.
    if ! command -v npm >/dev/null 2>&1; then
        echo "[INFO] npm not found; installing it for the Hermes front end..."
        sudo apt-get update -qq && sudo apt-get install -y -qq npm \
            || echo "[WARN] npm install failed; the Hermes TUI may be degraded."
    fi
    curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-setup
    echo "[OK] Hermes agent installed."
else
    echo "[INFO] Hermes agent already available. Skipping."
fi

# Second chown, after the installer has run: the install script can create
# files under $HOME/.hermes as root when invoked through sudo, which then makes
# every later `hermes config set` fail on permissions.
sudo chown -R $(whoami):$(whoami) "$HOME/.hermes"

echo "[INFO] Applying local backend configuration..."
hermes config set model.provider custom
hermes config set model.base_url "http://localhost:$VLLM_HERMES_PORT/v1"
hermes config set model.default "$HERMES_MODEL"
hermes config set compression.enabled false
hermes config set model.max_tokens 16384
hermes config set terminal.cwd "$WORKSPACE_DIR"
hermes config set tool_output.max_bytes 150000
hermes config set tool_output.max_lines 5000
hermes config set tool_output.max_line_length 5000

# ===========================================================================
# Playwright and browser dependencies (browser-driving Hermes tools)
# ===========================================================================
# Two things worth calling out here:
#   * Installing only the pip package without running `playwright install`
#     leaves the browser binary missing, so any browser tool fails at first
#     use. The Chromium download is done here so the success message is earned.
#   * The [OK] message is emitted only after a post-install import check, not
#     unconditionally.
# Locate the Hermes venv rather than assuming a path.
#
# The Hermes installer links the binary into /usr/local/bin and installs the
# code to /usr/local/lib/hermes-agent, not $HOME/.hermes/hermes-agent. On a
# root install (the workshop path) $HOME/.hermes/hermes-agent/venv does not
# exist at all, so a hardcoded path would install telemetry into a venv Hermes
# never runs, leaving the agent emitting no traces and the dashboard empty with
# no error.
#
# Resolve the venv from the `hermes` launcher itself, which is authoritative,
# and fall back to the known install locations.
find_hermes_venv_py() {
    local launcher py
    launcher="$(command -v hermes 2>/dev/null || true)"
    if [ -n "$launcher" ]; then
        # The launcher execs an absolute interpreter path; read it back.
        py="$(grep -oE '"/[^"]*/venv/bin/python"' "$launcher" 2>/dev/null \
              | head -1 | tr -d '"')"
        if [ -n "$py" ] && [ -x "$py" ]; then
            echo "$py"
            return 0
        fi
    fi
    for cand in \
        /usr/local/lib/hermes-agent/venv/bin/python \
        "$HOME/.hermes/hermes-agent/venv/bin/python" \
        /opt/hermes-agent/venv/bin/python; do
        if [ -x "$cand" ]; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

HERMES_VENV_PY="$(find_hermes_venv_py || true)"
if [ -n "$HERMES_VENV_PY" ]; then
    echo "[OK] Hermes venv: $HERMES_VENV_PY"
else
    echo "[WARN] Could not locate the Hermes venv."
    echo "       Telemetry and Playwright steps will be skipped, and the"
    echo "       profiling dashboard will have no data to display."
fi
if [ -n "$HERMES_VENV_PY" ] && [ -x "$HERMES_VENV_PY" ]; then
    if "$HERMES_VENV_PY" -c "import playwright" >/dev/null 2>&1; then
        echo "[OK] Playwright is already installed."
    else
        echo "[INFO] Installing Playwright..."
        # A killed install leaves this lock behind and every later attempt
        # blocks on it forever.
        rm -rf "$HOME/.cache/ms-playwright/__dirlock"
        "$HERMES_VENV_PY" -m pip install -q playwright \
            && "$HERMES_VENV_PY" -m playwright install chromium \
            || echo "[WARN] Playwright setup failed; browser tools unavailable."
        if "$HERMES_VENV_PY" -c "import playwright" >/dev/null 2>&1; then
            echo "[OK] Playwright installed."
        else
            echo "[WARN] Playwright still not importable after install."
        fi
    fi
else
    echo "[WARN] Hermes venv not found at $HERMES_VENV_PY; skipping Playwright."
fi

# ===========================================================================
# Hermes OpenTelemetry Plugin & Patch Setup
# ===========================================================================
echo "[INFO] Installing Hermes OpenTelemetry plugin..."

rm -rf "$HOME/.hermes/plugins/hermes_otel"
mkdir -p "$HOME/.hermes/plugins"
git clone https://github.com/briancaffey/hermes-otel.git "$HOME/.hermes/plugins/hermes_otel"

cd "$HOME/.hermes/plugins/hermes_otel"
git fetch origin

echo "[INFO] Applying advanced profiling patch..."
# Two patches, applied together in order: the base patch adds full-session
# CPU/GPU/tool CSV capture, and the addon (layered on top of it) adds MLflow
# run tracking + vLLM prefix-cache metrics. Both are generated against the
# same base commit in the hermes-otel repo, so if HEAD has since moved past
# it and no longer applies cleanly, reset to that exact commit first.
PATCH_FILE_GRAPH="$UTILS_DIR/hermes_advanced_profiling.patch"
PATCH_FILE_VLLM="$UTILS_DIR/hermes_vllm_addon.patch"

if [ ! -f "$PATCH_FILE_GRAPH" ] || [ ! -f "$PATCH_FILE_VLLM" ]; then
    echo "[ERROR] One or both patch files not found ($PATCH_FILE_GRAPH, $PATCH_FILE_VLLM); profiling not installed."
elif git apply --check "$PATCH_FILE_GRAPH" >/dev/null 2>&1 && git apply --check "$PATCH_FILE_VLLM" >/dev/null 2>&1; then
    git apply "$PATCH_FILE_GRAPH"
    git apply "$PATCH_FILE_VLLM"
    echo "[OK] Advanced profiling patches applied cleanly."
else
    echo "[WARN] Patches did not apply cleanly on current HEAD. Resetting to target commit 7497441..."
    git stash --include-untracked
    git checkout 7497441ccf156b9ed1f009fefe08935925bd7b42
    git apply "$PATCH_FILE_GRAPH"
    git apply "$PATCH_FILE_VLLM"
    echo "[OK] Reset and applied patches successfully."
fi

# Install the plugin package in editable mode using standard python/pip.
# Same PEP 668 opt-out as the MLflow install above; PIP_SYS_FLAGS is empty on
# interpreters that are not externally-managed.
echo "[INFO] Installing plugin package in editable mode..."
python3 -m pip install -q $PIP_SYS_FLAGS $PIP_SHADOW_DEBIAN -e .

# An editable install that silently no-ops leaves the agent running with no
# telemetry plugin and no error, so assert the package actually imports.
if ! python3 -c "import hermes_otel" 2>/dev/null; then
    echo "[WARN] hermes_otel is not importable from $(command -v python3) after the editable install."
    echo "       Telemetry may not be exported. Check the pip output above."
fi

cd "$WORKSPACE_DIR"

# Write the plugin config
cat << 'EOF' > "$HOME/.hermes/plugins/hermes_otel/config.yaml"
enabled: true
force_flush_on_session_end: true
capture_previews: true
capture_full_prompts: true
capture_full_responses: true
host_metrics: true
host_metrics_gpu: amd
host_metrics_interval_ms: 100
backends:
  - type: otlp
    name: mlflow
    endpoint: http://127.0.0.1:5004/v1/traces
    metrics: false
    logs: false
    headers:
      x-mlflow-experiment-id: "0"
EOF

hermes plugins enable hermes_otel --allow-tool-override

# Everything below MUST go into the venv Hermes actually runs. Using a
# hardcoded $HOME path here silently installed nothing on a root install and
# left the dashboard empty. See find_hermes_venv_py above.
if [ -z "$HERMES_VENV_PY" ] || [ ! -x "$HERMES_VENV_PY" ]; then
    echo "[FATAL] Hermes venv not found, so telemetry cannot be installed."
    echo "        The profiling dashboard would render empty with no error."
    echo "        Install Hermes first, then re-run this script."
    exit 1
fi

# The Hermes venv is created by `uv` and ships without pip, so every
# `-m pip install` into it fails with "No module named pip". Bootstrap pip
# first, and do not swallow the result: if pip cannot be installed here, none
# of the telemetry packages below land and the dashboard ends up empty with no
# visible error.
if ! "$HERMES_VENV_PY" -m pip --version >/dev/null 2>&1; then
    echo "[INFO] Hermes venv has no pip (uv-created); bootstrapping..."
    "$HERMES_VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
fi
if ! "$HERMES_VENV_PY" -m pip --version >/dev/null 2>&1; then
    echo "[FATAL] Could not bootstrap pip inside the Hermes venv:"
    echo "        $HERMES_VENV_PY"
    echo "        Telemetry cannot be installed and the profiling dashboard"
    echo "        would render empty. Refusing to continue."
    exit 1
fi
echo "[OK] Hermes venv pip: $("$HERMES_VENV_PY" -m pip --version 2>&1 | head -1)"
# Dependencies for the patched hermes-otel plugin, inside the Hermes venv.
# GPU numbers come from amdsmi queried directly in-process (gpu_probe.py /
# host_metrics.py), not from scraping an external AMD Device Metrics Exporter
# container over HTTP. CPU numbers come from `psutil`. `requests` is needed
# separately for the vLLM prefix-cache metrics feature (mlflow_hooks.py scrapes
# vLLM's own /metrics endpoint), unrelated to GPU readings.
#
# amdsmi is intentionally left unpinned: it ships with the ROCm stack and
# should match whatever ROCm version is already on this host rather than a
# hardcoded version here.
#
# psutil is installed with --no-deps deliberately: it is a leaf dependency and
# this keeps pip from touching anything else already resolved in the venv.
"$HERMES_VENV_PY" -m pip install -q \
  opentelemetry-api==1.42.1 opentelemetry-sdk==1.42.1 \
  opentelemetry-exporter-otlp-proto-http==1.42.1
"$HERMES_VENV_PY" -m pip install -q --no-deps psutil
"$HERMES_VENV_PY" -m pip install -q amdsmi
"$HERMES_VENV_PY" -m pip install -q matplotlib
"$HERMES_VENV_PY" -m pip install -q mlflow==3.13.0 requests

# The plugin package itself must also be importable from the Hermes venv, not
# just from the system interpreter, or the agent loads no telemetry backend.
"$HERMES_VENV_PY" -m pip install -q $PIP_SHADOW_DEBIAN \
  -e "$HOME/.hermes/plugins/hermes_otel" 2>/dev/null \
  || "$HERMES_VENV_PY" -m pip install -q -e "$HOME/.hermes/plugins/hermes_otel"

# Prove the plugin's imports actually resolve, instead of trusting pip's exit
# code. A missing exporter here is the failure that leaves the dashboard silently
# empty later.
"$HERMES_VENV_PY" - <<'PYCHECK'
import sys
missing = []
for mod in ("opentelemetry.sdk",
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            "psutil", "amdsmi", "requests", "mlflow", "hermes_otel"):
    try:
        __import__(mod)
    except Exception as exc:            # noqa: BLE001
        missing.append(f"{mod} ({exc.__class__.__name__})")
if missing:
    # Deliberately FATAL, not a warning: a missing dependency here means the
    # agent emits no traces and the dashboard renders empty with nothing in any
    # log to explain it, so stop rather than continuing to "[OK] Setup complete".
    print("[FATAL] Hermes venv is missing: " + ", ".join(missing))
    print("[FATAL] The agent would emit no telemetry and the profiling")
    print("        dashboard would render empty. Refusing to continue.")
    sys.exit(1)
print("[OK] Hermes venv telemetry dependencies import cleanly.")
PYCHECK
# This script does not use `set -e`, so the heredoc's exit status must be
# checked explicitly. Without this check the exit 1 above is discarded and the
# run continues to "[OK] Setup complete" with no telemetry installed.
if [ $? -ne 0 ]; then
    echo "[FATAL] Aborting: Hermes telemetry dependencies are not installed."
    exit 1
fi
echo "[INFO] MLflow tracking available at http://${SYSTEM_IP}:5004/"

cat << 'EOF' >> "$HOME/.hermes/.env"
HERMES_CSV_DUMP=true
HERMES_PLOT_PROFILING=true
HERMES_VLLM_CACHE_METRICS=true
HERMES_VLLM_PORT=8001
MLFLOW_TRACKING_URI=http://127.0.0.1:5004
MLFLOW_EXPERIMENT_NAME=Default
MLFLOW_RUN_NAME=Hermes_Profiling_{session_id}
EOF

echo "HERMES_PROFILING_OUTPUT_DIR=${WORKSPACE_DIR}/outputs" >> "$HOME/.hermes/.env"

# ===========================================================================
# Kokoro TTS server
# ===========================================================================
echo "[INFO] Setting up the Kokoro TTS server environment ($KOKORO_ENV)..."
if [ ! -d "$KOKORO_ENV" ]; then
    echo "[INFO] Creating Python venv at $KOKORO_ENV..."
    python3 -m venv "$KOKORO_ENV"
fi
"$KOKORO_ENV/bin/python" -m pip install -q --upgrade pip
echo "[INFO] Installing PyTorch (ROCm 7.2)..."
"$KOKORO_ENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
echo "[INFO] Installing kokoro, soundfile, fastapi, uvicorn, streamlit..."
"$KOKORO_ENV/bin/pip" install kokoro soundfile fastapi uvicorn

# Install the dashboard's dependencies from utils/requirements.txt rather than
# naming streamlit alone.
#
# Installing only streamlit leaves plotly absent, so utils/hermes_profiler.py
# fails on `import plotly.graph_objects` and the dashboard renders a bare
# ModuleNotFoundError traceback. A health check alone would still report the
# dashboard up, because /_stcore/health returns 200 for a crashed app: the
# Streamlit server is alive even when the script inside it is not.
if [ -f "$UTILS_DIR/requirements.txt" ]; then
    "$KOKORO_ENV/bin/pip" install -q -r "$UTILS_DIR/requirements.txt"
else
    echo "[WARN] $UTILS_DIR/requirements.txt not found; installing known deps."
    "$KOKORO_ENV/bin/pip" install -q 'streamlit>=1.30' 'plotly>=5.18' 'pandas>=2.0' mlflow==3.13.0
fi

# Assert every module the dashboard imports at top level actually resolves.
# pip's exit code is not evidence the app can start.
"$KOKORO_ENV/bin/python" - <<'PYDASH'
import sys
missing = []
for mod in ("streamlit", "plotly", "plotly.graph_objects", "pandas", "mlflow"):
    try:
        __import__(mod)
    except Exception as exc:            # noqa: BLE001
        missing.append(f"{mod} ({exc.__class__.__name__})")
if missing:
    print("[FATAL] Dashboard dependencies missing: " + ", ".join(missing))
    sys.exit(1)
print("[OK] Dashboard dependencies import cleanly.")
PYDASH
if [ $? -ne 0 ]; then
    echo "[FATAL] Aborting: the telemetry dashboard cannot start."
    exit 1
fi

# MIOpen needs this lock directory to persist its kernel DB and avoid errors on
# new shapes; create it before the server starts.
mkdir -p "$HOME/.config/miopen/miopen-lockfiles"

# ---------------------------------------------------------------------------
# MIOpen JIT headers
# ---------------------------------------------------------------------------
# Kokoro's text encoder runs an LSTM, and MIOpen compiles that kernel at runtime
# with HIPRTC. That compile needs ROCm headers on disk, not just the runtime
# libraries. Some AMD Dev Cloud ROCm images ship the libraries but omit the
# header trees, and the resulting failure is misleading:
#
#   RuntimeError: miopenStatusUnknownError        (inside _VF.lstm)
#
# with no mention of a missing file. Other GPU operations still succeed
# (torch.cuda.is_available(), a matmul, a plain torch.nn.LSTM on GPU), so only
# the JIT-compiled kernel fails and it presents as a Kokoro bug rather than a
# missing header.
#
# The ROCm docker images already on these hosts carry the full header tree, so
# extract from one instead of relying on an apt repo (Dev Cloud images have no
# ROCm apt source configured, making `apt-get install rocrand-dev` a silent
# no-op). A stopped container is enough; no GPU and no run required.
ensure_miopen_jit_headers() {
    if [ -f /opt/rocm/include/rocrand/rocrand_xorwow.h ] \
       && [ -f /opt/rocm/include/hip/hip_runtime.h ]; then
        echo "[OK] MIOpen JIT headers present."
        return 0
    fi

    echo "[INFO] MIOpen JIT headers missing; extracting from a local ROCm image..."
    local src=""
    for cand in "$IMAGE_NAME" "rocm:latest"; do
        [ -n "$cand" ] || continue
        if sudo docker image inspect "$cand" >/dev/null 2>&1; then
            src="$cand"
            break
        fi
    done

    if [ -z "$src" ]; then
        echo "[WARN] No local ROCm image to extract headers from."
        echo "       Kokoro may fail with 'miopenStatusUnknownError' in _VF.lstm."
        return 0
    fi

    local cid
    cid="$(sudo docker create "$src" 2>/dev/null)" || {
        echo "[WARN] Could not create a container from $src; skipping header extraction."
        return 0
    }
    for hdr in rocrand hiprand hip hsa half rocblas; do
        sudo docker cp "$cid:/opt/rocm/include/$hdr" /opt/rocm/include/ >/dev/null 2>&1 \
            && echo "  extracted $hdr" || true
    done
    sudo docker rm -f "$cid" >/dev/null 2>&1 || true

    if [ -f /opt/rocm/include/rocrand/rocrand_xorwow.h ] \
       && [ -f /opt/rocm/include/hip/hip_runtime.h ]; then
        echo "[OK] MIOpen JIT headers installed from $src."
    else
        echo "[WARN] Header extraction incomplete; Kokoro GPU synthesis may fail."
    fi
}

ensure_miopen_jit_headers

echo "[INFO] Launching Kokoro TTS server on port $KOKORO_PORT..."
KOKORO_PORT=$KOKORO_PORT "$KOKORO_ENV/bin/python" "$KOKORO_SERVER" > "$WORKSPACE_DIR/kokoro_server.log" 2>&1 &
KOKORO_PID=$!
echo "[INFO] Kokoro server started (PID $KOKORO_PID, logs: $WORKSPACE_DIR/kokoro_server.log)."

echo "[INFO] Waiting for Kokoro server /health on port $KOKORO_PORT..."
kokoro_ready=0
for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$KOKORO_PORT/health")
    code="${code:-000}"
    if [ "$code" -eq 200 ]; then
        echo "[OK] Kokoro TTS server is active (sequential + batched)."
        kokoro_ready=1
        break
    fi
    # Fail fast if the background process already died.
    if ! kill -0 "$KOKORO_PID" 2>/dev/null; then
        break
    fi
    sleep 3
done
if [ "$kokoro_ready" -ne 1 ]; then
    fail "Kokoro TTS server" "$WORKSPACE_DIR/kokoro_server.log"
fi

# ===========================================================================
# Telemetry dashboard (Streamlit)
# ===========================================================================
# Runs for the whole session; --server.address 0.0.0.0 makes it reachable from
# other machines.
DASHBOARD_APP="$UTILS_DIR/hermes_profiler.py"
if [ -f "$DASHBOARD_APP" ]; then
    # Streamlit is installed into the Kokoro venv ($KOKORO_ENV), not system-wide,
    # so a bare `streamlit` only resolves if that venv is on PATH. On a clean
    # host it is not, and the launch dies with "streamlit: command not found"
    # inside the redirected log, surfacing later as a "[FATAL] Streamlit
    # telemetry dashboard failed to start". Prefer the venv binary and fall
    # back to whatever is on PATH.
    STREAMLIT_BIN="$KOKORO_ENV/bin/streamlit"
    if [ ! -x "$STREAMLIT_BIN" ]; then
        STREAMLIT_BIN="$(command -v streamlit 2>/dev/null)"
    fi
    if [ -z "$STREAMLIT_BIN" ]; then
        echo "[FATAL] streamlit not found in $KOKORO_ENV/bin or on PATH."
        echo "        The venv install above should have provided it; check its output."
        exit 1
    fi
    echo "[INFO] Launching telemetry dashboard on port 8501 using $STREAMLIT_BIN..."

    # Streamlit resolves .streamlit/config.toml from the PROCESS CWD, not from
    # the directory of the script passed to `run`. The AMD theme lives in
    # $UTILS_DIR/.streamlit, so launch from there or the whole theme is silently
    # dropped with nothing in any log.
    (
        cd "$UTILS_DIR" || exit 1
        "$STREAMLIT_BIN" run "$DASHBOARD_APP" \
            --server.address 0.0.0.0 \
            --server.port 8501 \
            --server.headless true > "$WORKSPACE_DIR/streamlit_dashboard.log" 2>&1
    ) &
    STREAMLIT_PID=$!
    echo "[INFO] Dashboard started (PID $STREAMLIT_PID)."

    echo "[INFO] Waiting for Streamlit dashboard /_stcore/health on port 8501..."
    streamlit_ready=0
    for i in $(seq 1 30); do
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8501/_stcore/health")
        code="${code:-000}"
        if [ "$code" -eq 200 ]; then
            streamlit_ready=1
            break
        fi
        # Fail fast if the background process already died.
        if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
            break
        fi
        sleep 2
    done
    if [ "$streamlit_ready" -ne 1 ]; then
        fail "Streamlit telemetry dashboard" "$WORKSPACE_DIR/streamlit_dashboard.log"
    fi

    # /_stcore/health returning 200 only proves the Streamlit server is alive.
    # It returns 200 even when the app script raised on import and every visitor
    # sees a traceback (for example a missing plotly producing a
    # ModuleNotFoundError page).
    #
    # Parse the app's own top-level imports and confirm each one resolves in the
    # interpreter Streamlit runs under. That is what the health endpoint cannot
    # tell us.
    dash_bad="$("$KOKORO_ENV/bin/python" - "$DASHBOARD_APP" <<'PYPROBE'
import ast
import importlib.util
import sys

path = sys.argv[1]
try:
    tree = ast.parse(open(path).read())
except Exception as exc:                # noqa: BLE001
    print(f"UNPARSEABLE:{exc.__class__.__name__}")
    raise SystemExit(0)

mods = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for a in node.names:
            mods.add(a.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        mods.add(node.module.split(".")[0])

bad = []
for m in sorted(mods):
    if m in sys.builtin_module_names:
        continue
    try:
        if importlib.util.find_spec(m) is None:
            bad.append(m)
    except Exception:                   # noqa: BLE001
        bad.append(m)
print(",".join(bad))
PYPROBE
)"
    if [ -n "$dash_bad" ]; then
        echo "[FATAL] The dashboard is serving an error page."
        echo "        $DASHBOARD_APP imports modules that are not installed in"
        echo "        $KOKORO_ENV: $dash_bad"
        echo "        Note /_stcore/health still returns 200, which is why this"
        echo "        is checked separately."
        exit 1
    fi
    echo "[OK] Streamlit dashboard is up and every app import resolves."
else
    fail "Streamlit telemetry dashboard" ""
fi

echo -e "\n========================================================================="
echo "[OK] Setup complete."
echo "  vLLM endpoint:        http://${SYSTEM_IP}:$VLLM_HERMES_PORT"
echo "  Kokoro TTS server:    http://${SYSTEM_IP}:$KOKORO_PORT"
echo "  MLflow tracking:      http://${SYSTEM_IP}:5004"
echo "  Telemetry dashboard:  http://${SYSTEM_IP}:8501"
echo "========================================================================="
echo "[INFO] Holding the session open. Press Ctrl+C to stop all services and exit."

# Keep the shell process alive so the cleanup trap stays active.
while true; do
    sleep 60
done
