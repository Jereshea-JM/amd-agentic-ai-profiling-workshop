#!/bin/bash

# ===========================================================================
# Configuration
# ===========================================================================
HERMES_WORKSPACE_DIR="$HOME/.hermes/workspace"
mkdir -p "$HERMES_WORKSPACE_DIR"
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# Path anchoring: this script lives in utils/, so resolve its own siblings
# (clear_cache.sh, kokoro_server.py, hermes_profiler.py, the profiling patch)
# relative to the script itself. WORKSPACE_DIR stays the REPO ROOT because that
# is where the notebook, the env/ venv, input_text.txt and outputs/ live, and it
# is what the agent's terminal.cwd is pointed at. This makes the script safe to
# invoke from anywhere, for example `bash utils/helper.sh` from the repo root.
UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$UTILS_DIR/.." && pwd)"
cd "$WORKSPACE_DIR"

export PATH="$HOME/.local/bin:$PATH"

# Upstream tts-aug18 added `source ~/.bashrc` and an unconditional
# `sudo chown -R $USER:$USER $HOME/.cache/` here. Both are ported, but guarded:
#   * ~/.bashrc short-circuits on non-interactive shells on Debian/Ubuntu, and
#     `set -u` inside a user's rc file would kill this script, so it is sourced
#     defensively and only when it exists.
#   * The chown fixes a real failure: on a fresh machine an earlier root-run
#     step can leave $HOME/.cache root-owned, after which the HF download and
#     the Playwright install both fail with EACCES. It is skipped when the cache
#     is already owned correctly, so the common path costs nothing and the
#     script still works where sudo is unavailable (for example in the
#     container, which already runs as root).
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

HERMES_GPU="0"   # Muse-Glimmer-30B runs on GPU 0

# vLLM image: built locally from the ROCm nightly with PR #51655 (Muse-Glimmer
# support) overlaid. Built once below if not already present.
IMAGE_NAME="vllm-muse-glimmer:rocm"
VLLM_HERMES_PORT=8001
VLLM_DEVICE_METRICS_EXPORTER_PORT=5050

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

    echo "[INFO] Stopping device-metrics-exporter container..."
    sudo docker stop device-metrics-exporter >/dev/null 2>&1
    sudo docker rm device-metrics-exporter >/dev/null 2>&1

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
sudo docker rm -f hermes_service device-metrics-exporter >/dev/null 2>&1

# Build the image locally if it is missing: clone the PR, overlay its Python
# files onto the nightly's installed vLLM (no kernel compile), then commit.
BASE_IMAGE="vllm/vllm-openai-rocm:nightly"
if [ -z "$(sudo docker images -q $IMAGE_NAME)" ]; then
    echo "[INFO] Image $IMAGE_NAME not found. Building from $BASE_IMAGE + PR #51655..."
    sudo docker pull "$BASE_IMAGE"
    sudo docker rm -f muse_build >/dev/null 2>&1

    # Overlay PR #51655's Python files and verify the parsers register before
    # committing.
    sudo docker run --name muse_build \
        --device=/dev/kfd --device=/dev/dri \
        --security-opt seccomp=unconfined --group-add video --privileged \
        --entrypoint /bin/bash "$BASE_IMAGE" -c '
            set -e
            apt-get update && apt-get install -y git rsync
            git clone https://github.com/vllm-project/vllm.git /tmp/vllm-src
            cd /tmp/vllm-src
            git fetch origin pull/51655/head:muse
            git checkout muse
            cd /root
            VLLM_PKG=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -1)
            echo "Overlaying PR #51655 python files onto: $VLLM_PKG"
            rsync -a --include="*/" --include="*.py" --exclude="*" /tmp/vllm-src/vllm/ "$VLLM_PKG/"
            python3 -c "from vllm.tool_parsers import ToolParserManager; assert ToolParserManager.get_tool_parser(\"muse_glimmer\"); print(\"tool parser OK\")"
            python3 -c "from vllm.reasoning import ReasoningParserManager; assert ReasoningParserManager.get_reasoning_parser(\"muse_glimmer\"); print(\"reasoning parser OK\")"
        '
    BUILD_RC=$?
    if [ "$BUILD_RC" -ne 0 ]; then
        echo "[ERROR] Build of $IMAGE_NAME failed (overlay step). See output above."
        sudo docker rm -f muse_build >/dev/null 2>&1
        exit 1
    fi

    echo "[INFO] Committing patched container to image $IMAGE_NAME..."
    sudo docker commit muse_build "$IMAGE_NAME"
    sudo docker rm -f muse_build >/dev/null 2>&1
    echo "[OK] Built $IMAGE_NAME"
else
    echo "[OK] Image $IMAGE_NAME found locally. Skipping build."
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

echo "[INFO] Launching device-metrics-exporter to track GPU usage..."
sudo docker run -d \
    --device=/dev/dri \
    --device=/dev/kfd \
    -v /sys:/sys:ro \
    -p $VLLM_DEVICE_METRICS_EXPORTER_PORT:5000 \
    --name device-metrics-exporter \
    rocm/device-metrics-exporter:v1.5.0

# The exporter serves Prometheus metrics at /metrics (not /v1/models), so it
# needs its own readiness check rather than wait_for_vllm_readiness.
echo "[INFO] Waiting for device-metrics-exporter on port $VLLM_DEVICE_METRICS_EXPORTER_PORT..."
exporter_ready=0
for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" \
        "http://localhost:$VLLM_DEVICE_METRICS_EXPORTER_PORT/metrics")
    code="${code:-000}"
    if [ "$code" -eq 200 ]; then
        echo "[OK] device-metrics-exporter is serving metrics."
        exporter_ready=1
        break
    fi
    sleep 2
done
if [ "$exporter_ready" -ne 1 ]; then
    echo "[WARN] device-metrics-exporter did not respond on /metrics; GPU columns may be empty."
fi

# ===========================================================================
# Hermes toolchain and MLflow integration
# ===========================================================================
echo "[INFO] Installing MLflow and OpenTelemetry dependencies..."

# Bootstrap a usable Python toolchain BEFORE anything tries to pip install.
#
# Several AMD Dev Cloud ROCm images (verified on rocm714-vllm-0.27.1-omni,
# Ubuntu 24.04, Python 3.12.3) ship with NO pip and NO ensurepip for the system
# interpreter, and they mark it PEP 668 externally-managed. Every `python3 -m
# pip` below then dies with "No module named pip". That failure used to surface
# far downstream as "[FATAL] MLflow server failed to start properly", whose log
# said only "No module named mlflow", which points at the wrong problem
# entirely.
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
# plain `pip install` is refused. These are throwaway workshop boxes and the
# script already owns the system Python, so opt out explicitly rather than
# letting the install fail.
PIP_SYS_FLAGS=""
if python3 -c "import sys,sysconfig,os; \
sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path('stdlib'), \
'EXTERNALLY-MANAGED')) else 1)" 2>/dev/null; then
    PIP_SYS_FLAGS="--break-system-packages"
    echo "[INFO] Interpreter is PEP 668 externally-managed; using --break-system-packages."
fi

# Distro-installed Python packages carry no RECORD file, so when pip needs to
# upgrade one to satisfy a dependency it cannot uninstall it and aborts the
# WHOLE transaction:
#
#   ERROR: Cannot uninstall typing_extensions 4.10.0, RECORD file not found.
#          Hint: The package was installed by debian.
#
# Verified on rocm714-vllm-0.27.1-omni: mlflow 3.13.0 pulls a newer
# typing_extensions than the apt-shipped 4.10.0. --ignore-installed on just the
# offending names lets pip shadow them in site-packages without trying to
# remove the apt copy. Scoped deliberately: a blanket --ignore-installed would
# redownload the entire dependency tree.
PIP_SHADOW_DEBIAN="--ignore-installed typing_extensions"

# opentelemetry-exporter-otlp-proto-http is required: the hermes-otel plugin
# ships traces to MLflow over the OTLP/HTTP protobuf endpoint. Without it the
# plugin loads and prints its banner but exports nothing, so the dashboard sits
# empty with no error. Added in upstream tts-aug18.
python3 -m pip install -q $PIP_SYS_FLAGS $PIP_SHADOW_DEBIAN \
  mlflow==3.13.0 opentelemetry-sdk==1.42.1 \
  opentelemetry-exporter-otlp-proto-http==1.42.1 \
  "anyio<4.5.0"

# Fail HERE with an accurate message rather than 200 lines later as a confusing
# "MLflow server failed to start" whose log only says "No module named mlflow".
# Check the OTLP exporter too: without it the plugin exports nothing silently.
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
    # provisioned first. Ported from upstream tts-aug18, made conditional so a
    # machine that already has npm (and the container image) does not pay for an
    # apt round-trip, and a failure here does not abort the whole setup.
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
# every later `hermes config set` fail on permissions. Ported from tts-aug18.
sudo chown -R $(whoami):$(whoami) "$HOME/.hermes"

echo "[INFO] Applying local backend configuration..."
hermes config set model.provider custom
hermes config set model.base_url "http://localhost:$VLLM_HERMES_PORT/v1"
hermes config set model.default "$HERMES_MODEL"
hermes config set compression.enabled false
# 16384, raised from 8192: the workshop passage is now ~8,450 characters, and
# at 8192 the agent's replies were being truncated mid-run.
hermes config set model.max_tokens 16384
hermes config set terminal.cwd "$WORKSPACE_DIR"
hermes config set tool_output.max_bytes 150000
hermes config set tool_output.max_lines 5000
hermes config set tool_output.max_line_length 5000

# ===========================================================================
# Playwright and browser dependencies (browser-driving Hermes tools)
# ===========================================================================
# Ported from upstream tts-aug18. Two corrections were needed:
#   * Upstream installs only the pip package and never runs `playwright
#     install`, so the browser binary is missing and any browser tool fails at
#     first use, after the script has already printed "installed successfully".
#     The Chromium download is done here so the success message is earned.
#   * Upstream reports [OK] unconditionally. Here the message is emitted only
#     after a real post-install import check.
# Locate the Hermes venv rather than assuming a path.
#
# The Hermes installer links the binary into /usr/local/bin and installs the
# code to /usr/local/lib/hermes-agent, NOT $HOME/.hermes/hermes-agent. On a
# root install (the workshop path) $HOME/.hermes/hermes-agent/venv does not
# exist at all.
#
# Verified on a clean MI300X run of merged main on 2026-08-20: the hardcoded
# path caused FIVE consecutive silent failures, all buried at log line ~2472
# while setup still printed "[OK] Setup complete":
#
#   utils/helper.sh: line 538: /root/.hermes/hermes-agent/venv/bin/python: No such file or directory
#   ... lines 549, 552, 553, 558 identical
#
# The consequence was invisible: mlflow, opentelemetry-sdk and the hermes_otel
# plugin were never installed into the venv Hermes actually runs, so the agent
# emitted no traces, MLflow held 0 runs and 0 traces, no profiling CSVs were
# written, and the telemetry dashboard rendered empty with no error anywhere.
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
git fetch origin --tags --depth=1
git checkout hermes-otel-v0.10.0

# Apply the advanced profiling patch once while inside the plugin directory
echo "[INFO] Applying advanced profiling patch..."
PATCH_FILE="$UTILS_DIR/hermes_advanced_profiling.patch"

if [ ! -f "$PATCH_FILE" ]; then
    echo "[ERROR] Patch file not found at $PATCH_FILE; profiling not installed."
elif git apply --check "$PATCH_FILE" >/dev/null 2>&1; then
    git apply "$PATCH_FILE"
    echo "[OK] Advanced profiling patch applied."
elif git apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "[INFO] Patch already applied. Skipping."
else
    echo "[WARN] Patch did not apply cleanly (wrong plugin version or conflict)."
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

# The Hermes venv is created by `uv` and ships WITHOUT pip (verified on a clean
# MI300X run 2026-08-20: pyvenv.cfg shows `uv = 0.12.5`, and every
# `-m pip install` failed with "No module named pip"). Bootstrap it, and do NOT
# swallow the result: if pip cannot be installed here, none of the telemetry
# packages below land and the dashboard ends up empty with no visible error.
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
# Upstream tts-aug18 trimmed pyrsmi, amdsmi and cryptography from this line and
# the trim is correct: nothing in this repo imports them. The patched plugin
# gets GPU numbers by scraping the AMD Device Metrics Exporter over HTTP
# (`requests`), not through amdsmi/pyrsmi bindings, and its CPU numbers come
# from `psutil`. Carrying the extra wheels only risked pip resolving a
# conflicting transitive dependency into the Hermes venv.
#
# psutil is installed with --no-deps deliberately: it is a leaf dependency and
# this keeps pip from touching anything else already resolved in the venv.
"$HERMES_VENV_PY" -m pip install -q \
  opentelemetry-api==1.42.1 opentelemetry-sdk==1.42.1 \
  opentelemetry-exporter-otlp-proto-http==1.42.1
"$HERMES_VENV_PY" -m pip install -q --no-deps psutil
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
            "psutil", "requests", "mlflow", "hermes_otel"):
    try:
        __import__(mod)
    except Exception as exc:            # noqa: BLE001
        missing.append(f"{mod} ({exc.__class__.__name__})")
if missing:
    # Deliberately FATAL, not a warning. This used to sys.exit(0), so the run
    # continued to "[OK] Setup complete" while the agent emitted no traces at
    # all and the dashboard sat empty with nothing in any log to explain it.
    print("[FATAL] Hermes venv is missing: " + ", ".join(missing))
    print("[FATAL] The agent would emit no telemetry and the profiling")
    print("        dashboard would render empty. Refusing to continue.")
    sys.exit(1)
print("[OK] Hermes venv telemetry dependencies import cleanly.")
PYCHECK
# This script does NOT use `set -e`, so the heredoc's exit status must be
# checked explicitly. Without this the exit 1 above is discarded and the run
# continues to "[OK] Setup complete" with no telemetry, which is the exact
# failure being fixed.
if [ $? -ne 0 ]; then
    echo "[FATAL] Aborting: Hermes telemetry dependencies are not installed."
    exit 1
fi
echo "[INFO] MLflow tracking available at http://${SYSTEM_IP}:5004/"

cat << 'EOF' >> "$HOME/.hermes/.env"
# MLflow and vLLM observability configuration
MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=1
MLFLOW_TRACKING_URI=http://127.0.0.1:5004
MLFLOW_EXPERIMENT_NAME=Default
VLLM_HERMES_PORT=8001
MLFLOW_RUN_NAME=Hermes_Profiling
MLFLOW_KEEP_RUN_ACTIVE=false
MLFLOW_LOGGING_LEVEL=ERROR
MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT=1
HERMES_TOOL_PROFILING=1
HERMES_GPU_EXPORTER_URL=http://localhost:5050/metrics
HERMES_PROFILING_DEBUG=0
MLFLOW_DISABLE_TELEMETRY=true
HERMES_CPU_DEBUG=0
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
# Verified on a clean MI300X run 2026-08-20: installing only streamlit meant
# plotly was never present, so utils/hermes_profiler.py died on
# `import plotly.graph_objects` and the dashboard rendered a bare
# ModuleNotFoundError traceback. Setup still printed "[OK] Streamlit dashboard
# is up." because /_stcore/health returns 200 for a crashed app: the Streamlit
# server is alive, the script inside it is not.
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
# Kokoro's text encoder runs an LSTM, and MIOpen compiles that kernel at RUNTIME
# with HIPRTC. That compile needs ROCm HEADERS on disk, not just the runtime
# libraries. Several AMD Dev Cloud ROCm images ship the libraries but omit the
# header trees, and the failure is deeply misleading:
#
#   RuntimeError: miopenStatusUnknownError        (inside _VF.lstm)
#
# with no mention of a missing file. Everything else passes, verified on this
# image: torch.cuda.is_available() True, a 512x512 matmul fine, and a plain
# torch.nn.LSTM on GPU fine. Only the JIT-compiled kernel fails, so it looks
# like a Kokoro bug rather than a missing header.
#
# The ROCm docker images already on these hosts carry the full header tree, so
# extract from one instead of needing an apt repo (Dev Cloud images have no
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
    for cand in "$IMAGE_NAME" "$BASE_IMAGE" "rocm:latest"; do
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
    # Streamlit is installed INTO the Kokoro venv ($KOKORO_ENV), not system-wide,
    # so a bare `streamlit` only resolves if that venv happens to be on PATH.
    # On a clean host it is not, and the launch dies with
    # "streamlit: command not found" inside the redirected log, surfacing 30
    # seconds later as "[FATAL] Streamlit telemetry dashboard failed to start".
    # Prefer the venv binary and fall back to whatever is on PATH.
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

    # /_stcore/health returning 200 only proves the Streamlit SERVER is alive.
    # It returns 200 even when the app script raised on import and every visitor
    # sees a traceback. Verified 2026-08-20: a missing plotly produced a
    # ModuleNotFoundError page while this loop still printed [OK].
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
