#!/bin/bash

# ===========================================================================
# Configuration
# ===========================================================================
HERMES_WORKSPACE_DIR="$HOME/.hermes/workspace"
mkdir -p "$HERMES_WORKSPACE_DIR"
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
WORKSPACE_DIR=$(pwd)

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$HOME/.cache/huggingface"

HERMES_GPU="0"   # Muse-Glimmer-30B runs on GPU 0

# vLLM image: built locally from the ROCm nightly with PR #51655 (Muse-Glimmer
# support) overlaid. Built once below if not already present.
IMAGE_NAME="vllm-muse-glimmer:rocm"
VLLM_HERMES_PORT=8001
VLLM_DEVICE_METRICS_EXPORTER_PORT=5050

SYSTEM_IP=$(ip route get 1 2>/dev/null | awk '{print $7; exit}' || ip route get 8.8.8.8 | awk '{print $7; exit}')

# Clear caches before starting.
bash clear_cache.sh

HERMES_MODEL="meta-models/Muse-Glimmer-30B"

# Local Kokoro TTS server (sequential + batched inference modes).
KOKORO_PORT=8092
KOKORO_ENV="$WORKSPACE_DIR/env"
KOKORO_SERVER="$WORKSPACE_DIR/kokoro_server.py"

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
        status_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/v1/models || echo "000")

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
        "http://localhost:$VLLM_DEVICE_METRICS_EXPORTER_PORT/metrics" || echo "000")
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
python3 -m pip install -q mlflow==3.13.0 opentelemetry-sdk==1.42.1

echo "[INFO] Launching MLflow server on port 5004..."
python3 -m mlflow server \
  --host 0.0.0.0 \
  --port 5004 \
  --backend-store-uri sqlite:///mlflow.db \
  --allowed-hosts "*" > mlflow_server.log 2>&1 &

MLFLOW_PID=$!
echo "[INFO] MLflow server started (PID $MLFLOW_PID)."

echo "[INFO] Waiting for MLflow server /health on port 5004..."
mlflow_ready=0
for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:5004/health" || echo "000")
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
    curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-setup
    echo "[OK] Hermes agent installed."
else
    echo "[INFO] Hermes agent already available. Skipping."
fi

echo "[INFO] Applying local backend configuration..."
hermes config set model.provider custom
hermes config set model.base_url "http://localhost:$VLLM_HERMES_PORT/v1"
hermes config set model.default "$HERMES_MODEL"
hermes config set compression.enabled false
hermes config set model.max_tokens 8192
hermes config set terminal.cwd "$WORKSPACE_DIR"

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
PATCH_FILE="$WORKSPACE_DIR/hermes_advanced_profiling.patch"

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

# Install the plugin package in editable mode using standard python/pip
echo "[INFO] Installing plugin package in editable mode..."
python3 -m pip install -e .

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

$HOME/.hermes/hermes-agent/venv/bin/python -m ensurepip --upgrade
$HOME/.hermes/hermes-agent/venv/bin/python -m pip install -q opentelemetry-api==1.42.1 opentelemetry-sdk==1.42.1 opentelemetry-exporter-otlp-proto-http==1.42.1 pyrsmi==1.1.0 amdsmi==7.0.2 mlflow==3.13.0 psutil requests cryptography
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

# Deploy the custom Kokoro TTS tool into the Hermes tools directory.
echo "[INFO] Deploying custom Kokoro TTS tool..."
HERMES_TOOLS_DIR="$HOME/.hermes/hermes-agent/tools"
mkdir -p "$HERMES_TOOLS_DIR"
if [ -f "$WORKSPACE_DIR/custom_tools/kokoro_tts_tool.py" ]; then
    cp "$WORKSPACE_DIR/custom_tools/kokoro_tts_tool.py" "$HERMES_TOOLS_DIR/"
    echo "[OK] Copied kokoro_tts_tool.py -> $HERMES_TOOLS_DIR"
else
    echo "[WARN] custom_tools/kokoro_tts_tool.py not found."
fi

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
"$KOKORO_ENV/bin/pip" install 'streamlit>=1.30'

# MIOpen needs this lock directory to persist its kernel DB and avoid errors on
# new shapes; create it before the server starts.
mkdir -p "$HOME/.config/miopen/miopen-lockfiles"

echo "[INFO] Launching Kokoro TTS server on port $KOKORO_PORT..."
KOKORO_PORT=$KOKORO_PORT "$KOKORO_ENV/bin/python" "$KOKORO_SERVER" > "$WORKSPACE_DIR/kokoro_server.log" 2>&1 &
KOKORO_PID=$!
echo "[INFO] Kokoro server started (PID $KOKORO_PID, logs: $WORKSPACE_DIR/kokoro_server.log)."

echo "[INFO] Waiting for Kokoro server /health on port $KOKORO_PORT..."
kokoro_ready=0
for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$KOKORO_PORT/health" || echo "000")
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
DASHBOARD_APP="$WORKSPACE_DIR/hermes_profiler.py"
if [ -f "$DASHBOARD_APP" ]; then
    echo "[INFO] Launching telemetry dashboard on port 8501..."
    streamlit run "$DASHBOARD_APP" \
        --server.address 0.0.0.0 \
        --server.port 8501 \
        --server.headless true > streamlit_dashboard.log 2>&1 &
    STREAMLIT_PID=$!
    echo "[INFO] Dashboard started (PID $STREAMLIT_PID)."

    echo "[INFO] Waiting for Streamlit dashboard /_stcore/health on port 8501..."
    streamlit_ready=0
    for i in $(seq 1 30); do
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8501/_stcore/health" || echo "000")
        if [ "$code" -eq 200 ]; then
            echo "[OK] Streamlit dashboard is up."
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
