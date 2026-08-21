#!/bin/bash
# =============================================================================
# AMD Agentic AI Profiling Workshop: in-container service manager
# =============================================================================
# The in-container equivalent of utils/helper.sh.
#
# Difference from helper.sh: helper.sh runs on a bare host and launches vLLM and
# the metrics exporter as SIBLING Docker containers. Inside this image there is
# no nested Docker, so every service is started here as a local process and the
# GPU is reached directly through the container's own /dev/kfd and /dev/dri.
#
# Services, in start order:
#   1. vLLM (Muse-Glimmer-30B)  :8001   the agent's model
#   2. MLflow tracking server   :5004   records traces and hardware metrics
#   3. Kokoro TTS server        :8092   local TTS on the MI300X
#   4. Streamlit dashboard      :8501   telemetry overview
#   5. JupyterLab               :8888   the workshop front door
#
# Usage:
#   serve       start everything and stay in the foreground (default)
#   bash        drop into a shell (services not started)
#   <command>   run an arbitrary command instead
# =============================================================================
set -uo pipefail

WORKSHOP_DIR="${WORKSHOP_DIR:-/workshop}"
UTILS_DIR="${WORKSHOP_DIR}/utils"
LOG_DIR="${WORKSHOP_DIR}/logs"
VLLM_HERMES_PORT="${VLLM_HERMES_PORT:-8001}"
KOKORO_PORT="${KOKORO_PORT:-8092}"
MLFLOW_PORT="${MLFLOW_PORT:-5004}"
# Device Metrics Exporter payload, copied from rocm/device-metrics-exporter by
# the Dockerfile. Overridable so a host-run exporter can still be used instead.
DME_DIR="${DME_DIR:-/root/amd-dme}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
HERMES_MODEL="${HERMES_MODEL:-meta-models/Muse-Glimmer-30B}"
HERMES_GPU="${HERMES_GPU:-0}"
# vLLM must download about 60 GB on a cold cache, so the default wait is long.
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-3600}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"
# vLLM defaults to 0.92, which fails to start whenever anything else already
# holds VRAM (another tenant, a stale process, or our own Kokoro server). The
# workshop only needs a 30B model plus a modest KV cache, so a lower default
# trades headroom we do not use for a container that actually starts. Override
# with -e GPU_MEMORY_UTILIZATION=... on a dedicated GPU.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"

mkdir -p "${LOG_DIR}" "${WORKSHOP_DIR}/outputs" \
         "${HOME}/.config/miopen/miopen-lockfiles"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() {
    echo "[ERROR] $1" >&2
    if [ -n "${2:-}" ] && [ -f "$2" ]; then
        echo "----- last 40 lines of $2 -----" >&2
        tail -n 40 "$2" >&2
    fi
    exit 1
}

PIDS=()
cleanup() {
    log "Shutting down services..."
    for pid in "${PIDS[@]:-}"; do
        [ -n "${pid}" ] && kill "${pid}" 2>/dev/null
    done
    wait 2>/dev/null
    log "Stopped."
}
trap cleanup EXIT INT TERM

# wait_for <url> <name> <timeout_s> <logfile> [accept_any_http]
# Polls until the endpoint answers. By default only HTTP 2xx/3xx counts as
# ready; pass accept_any_http=1 when any response proves the port is serving.
wait_for() {
    local url="$1" name="$2" timeout="$3" logfile="$4" any="${5:-0}"
    local watch_pid="${6:-}"
    local waited=0 code
    log "Waiting for ${name} (${url}, timeout ${timeout}s)..."
    while [ "${waited}" -lt "${timeout}" ]; do
        # NOTE: no `|| echo 000` here. curl already prints 000 via -w when it
        # cannot connect, so the fallback CONCATENATED a second 000 and produced
        # "000000". The any=1 branch below tests `code != "000"`, so "000000"
        # slipped through and a service was reported up when nothing answered.
        # Observed live 2026-08-21: "Metrics exporter is up (HTTP 000000) after 0s."
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${url}" 2>/dev/null)"
        code="${code:-000}"
        if [ "${any}" = "1" ] && [ "${code}" != "000" ]; then
            log "  ${name} is up (HTTP ${code}) after ${waited}s."
            return 0
        fi
        if [[ "${code}" =~ ^[23] ]]; then
            log "  ${name} is up (HTTP ${code}) after ${waited}s."
            return 0
        fi
        # Fail immediately if the process is already dead, rather than burning
        # the whole timeout waiting for a port that will never open.
        if [ -n "${watch_pid}" ] && ! kill -0 "${watch_pid}" 2>/dev/null; then
            fail "${name} exited before it became ready." "${logfile}"
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 60)) -eq 0 ]; then
            log "  still waiting for ${name}... ${waited}s (last HTTP ${code})"
        fi
    done
    fail "${name} did not become ready within ${timeout}s." "${logfile}"
}

start_services() {
    log "==================================================================="
    log " AMD Agentic AI Profiling Workshop"
    log "==================================================================="

    if [ ! -e /dev/kfd ]; then
        echo "[ERROR] /dev/kfd is missing: the container has no GPU access." >&2
        echo "        Re-run with: --device=/dev/kfd --device=/dev/dri" >&2
        echo "                     --security-opt seccomp=unconfined --group-add video" >&2
        exit 1
    fi
    log "GPU devices present. Detected:"
    amd-smi static 2>/dev/null | grep -m2 MARKET_NAME || log "  (amd-smi unavailable)"

    # Report VRAM already held by other processes. vLLM sizes its allocation
    # against FREE memory, so a busy GPU is the usual cause of a startup
    # failure, and a cryptic ValueError deep in the engine is a poor way to
    # discover that.
    if command -v amd-smi >/dev/null 2>&1; then
        local used_vram free_vram
        used_vram="$(amd-smi metric -m 2>/dev/null | grep -m1 'USED_VRAM' | awk '{print $2}')"
        free_vram="$(amd-smi metric -m 2>/dev/null | grep -m1 'FREE_VRAM' | awk '{print $2}')"
        if [ -n "${used_vram:-}" ]; then
            log "  VRAM in use by other processes: ${used_vram} MB (free: ${free_vram:-?} MB)"
            if [ "${used_vram}" -gt 2000 ] 2>/dev/null; then
                log "  NOTE: this GPU is not idle. Using --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}."
                log "        If vLLM still fails to start, lower it with -e GPU_MEMORY_UTILIZATION=0.70"
            fi
        fi
    fi

    # --- 0. AMD Device Metrics Exporter (embedded GPU metrics source) --------
    # The profiling poller scrapes HERMES_GPU_EXPORTER_URL for GPU utilization.
    # The exporter binaries were copied from rocm/device-metrics-exporter at
    # build time into /root/amd-dme, so no exporter needs to be running on the
    # Docker host. This mirrors the vendor entrypoint exactly: gpuagent reads
    # the GPU behind a unix socket with libamd_smi preloaded, then after a 10s
    # warm-up `server` publishes Prometheus metrics on :5000.
    log "Starting AMD Device Metrics Exporter on :5000..."
    (
        LD_PRELOAD="${DME_DIR}/lib/libamd_smi.so.26" \
            "${DME_DIR}/bin/gpuagent" -s /var/run/gpuagent.sock &
        sleep 10
        exec "${DME_DIR}/bin/server"
    ) > "${LOG_DIR}/exporter.log" 2>&1 &
    EXPORTER_PID=$!
    PIDS+=("${EXPORTER_PID}")
    # any=1: /metrics answers 200, but accept any HTTP response as "listening".
    wait_for "http://localhost:5000/metrics" "Metrics exporter" 120 \
             "${LOG_DIR}/exporter.log" 1 "${EXPORTER_PID}"

    # --- 1. vLLM -------------------------------------------------------------
    log "Starting vLLM (${HERMES_MODEL}) on port ${VLLM_HERMES_PORT}..."
    log "  First start downloads about 60 GB of weights; this takes a while."
    # No --max-model-len: use the model's native 131,072 token window, which is
    # what utils/helper.sh does on a bare host. Capping it lower is not just a
    # memory tweak: Hermes refuses to start against any model advertising less
    # than 64K context, so a smaller value breaks the agent outright.
    HIP_VISIBLE_DEVICES="${HERMES_GPU}" VLLM_ROCM_USE_AITER=1 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${HERMES_MODEL}" \
        --served-model-name "${HERMES_MODEL}" \
        --tool-call-parser muse_glimmer \
        --reasoning-parser muse_glimmer \
        --enable-auto-tool-choice \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --port "${VLLM_HERMES_PORT}" \
        --host 0.0.0.0 > "${LOG_DIR}/vllm.log" 2>&1 &
    VLLM_PID=$!
    PIDS+=("${VLLM_PID}")

    # --- 2. MLflow -----------------------------------------------------------
    log "Starting MLflow on port ${MLFLOW_PORT}..."
    mlflow server \
        --host 0.0.0.0 \
        --port "${MLFLOW_PORT}" \
        --backend-store-uri "sqlite:///${WORKSHOP_DIR}/mlflow.db" \
        --serve-artifacts \
        --artifacts-destination "${WORKSHOP_DIR}/mlflow_artifacts" \
        > "${LOG_DIR}/mlflow.log" 2>&1 &
    PIDS+=($!)
    wait_for "http://localhost:${MLFLOW_PORT}/health" "MLflow" 300 "${LOG_DIR}/mlflow.log"

    # --- 3. Kokoro TTS -------------------------------------------------------
    log "Starting Kokoro TTS server on port ${KOKORO_PORT}..."
    ( cd "${WORKSHOP_DIR}" && KOKORO_PORT="${KOKORO_PORT}" \
        python3 "${UTILS_DIR}/kokoro_server.py" > "${LOG_DIR}/kokoro.log" 2>&1 ) &
    PIDS+=($!)
    wait_for "http://localhost:${KOKORO_PORT}/health" "Kokoro TTS" 900 "${LOG_DIR}/kokoro.log"

    # --- 4. Telemetry dashboard ---------------------------------------------
    log "Starting telemetry dashboard on port ${DASHBOARD_PORT}..."
    # Streamlit resolves .streamlit/config.toml from the PROCESS CWD, not from
    # the script's directory. The config lives in utils/, so launching from
    # WORKSHOP_DIR silently drops the AMD theme (primaryColor falls back to
    # unset). Run from UTILS_DIR and point the app's own paths at WORKSHOP_DIR.
    ( cd "${UTILS_DIR}" && streamlit run "${UTILS_DIR}/hermes_profiler.py" \
        --server.address 0.0.0.0 \
        --server.port "${DASHBOARD_PORT}" \
        --server.headless true > "${LOG_DIR}/dashboard.log" 2>&1 ) &
    PIDS+=($!)
    wait_for "http://localhost:${DASHBOARD_PORT}/_stcore/health" \
             "Telemetry dashboard" 300 "${LOG_DIR}/dashboard.log"

    # --- 5. JupyterLab -------------------------------------------------------
    log "Starting JupyterLab on port ${JUPYTER_PORT}..."
    ( cd "${WORKSHOP_DIR}" && jupyter lab \
        --ip=0.0.0.0 --port="${JUPYTER_PORT}" --no-browser --allow-root \
        --ServerApp.token="${JUPYTER_TOKEN}" \
        --ServerApp.root_dir="${WORKSHOP_DIR}" \
        > "${LOG_DIR}/jupyter.log" 2>&1 ) &
    PIDS+=($!)
    wait_for "http://localhost:${JUPYTER_PORT}/api" "JupyterLab" 300 "${LOG_DIR}/jupyter.log"

    # --- 6. vLLM last: it is the slowest, everything else is already usable ---
    if ! wait_for "http://localhost:${VLLM_HERMES_PORT}/v1/models" \
             "vLLM (${HERMES_MODEL})" "${VLLM_READY_TIMEOUT}" \
             "${LOG_DIR}/vllm.log" 0 "${VLLM_PID}"; then
        exit 1
    fi
    # A dead vLLM process is a different failure from a slow one, so name it.
    if grep -q "less than desired GPU memory utilization" "${LOG_DIR}/vllm.log" 2>/dev/null; then
        echo "[ERROR] vLLM could not reserve enough VRAM." >&2
        echo "        Another process is holding GPU memory. Retry with a lower" >&2
        echo "        value, for example: -e GPU_MEMORY_UTILIZATION=0.70" >&2
        exit 1
    fi

    local ip
    ip="$(hostname -i 2>/dev/null | awk '{print $1}')"
    log "==================================================================="
    log " All services are ready."
    log "==================================================================="
    log "  JupyterLab (start here) : http://<host>:${JUPYTER_PORT}/lab/tree/tts.ipynb"
    log "  Telemetry dashboard     : http://<host>:${DASHBOARD_PORT}"
    log "  MLflow UI               : http://<host>:${MLFLOW_PORT}"
    log "  vLLM OpenAI API         : http://<host>:${VLLM_HERMES_PORT}/v1"
    log "  (container IP: ${ip:-unknown}; logs in ${LOG_DIR})"
    if [ -z "${JUPYTER_TOKEN}" ]; then
        log "  JupyterLab has no token. Set -e JUPYTER_TOKEN=... to require one."
    fi
    log "==================================================================="

    # Hold the container open, and exit if any service dies.
    while true; do
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                echo "[ERROR] A service (pid ${pid}) exited. Logs in ${LOG_DIR}." >&2
                for f in "${LOG_DIR}"/*.log; do
                    echo "----- tail ${f} -----" >&2
                    tail -n 20 "${f}" >&2
                done
                exit 1
            fi
        done
        sleep 10
    done
}

case "${1:-serve}" in
    serve) start_services ;;
    bash|sh) exec /bin/bash ;;
    *) exec "$@" ;;
esac
