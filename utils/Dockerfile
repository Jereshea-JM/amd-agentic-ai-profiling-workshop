# =============================================================================
# AMD Agentic AI Profiling Workshop
# =============================================================================
# All-in-one workshop image for AMD Instinct MI300X.
#
# Everything runs INSIDE this single container: vLLM (Muse-Glimmer-30B), the
# Hermes agent with the hermes-otel OpenTelemetry plugin, the Kokoro TTS server,
# MLflow, Grafana otel-lgtm, the Streamlit telemetry dashboard, and JupyterLab.
# There is no nested Docker, unlike utils/helper.sh which orchestrates sibling
# containers on a bare host. See utils/docker-entrypoint.sh for the in-container
# service manager.
#
# BUILD (from the repo root):
#   docker build -f utils/Dockerfile -t amd-agentic-ai-profiling:mi300x .
#
# RUN:
#   docker run -d --name amd-agentic-ai-profiling \
#     --device=/dev/kfd --device=/dev/dri \
#     --security-opt seccomp=unconfined --group-add video --ipc=host \
#     --shm-size 16G -p 8888:8888 -p 8501:8501 -p 5004:5004 \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     amd-agentic-ai-profiling:mi300x
#
# The model weights (about 60 GB) are downloaded on first start. Mount the
# Hugging Face cache as shown so the download is paid only once per host.
# =============================================================================

# The ROCm vLLM release image provides Python 3.12, PyTorch with HIP, and a
# working vLLM install (including Muse-Glimmer-30B support), so the Kokoro TTS
# server reuses that same PyTorch rather than installing a second multi-gigabyte
# copy.
# The Grafana otel-lgtm metrics backend is pulled in as a build stage so it ships
# inside this image and runs as a local process. utils/helper.sh runs the same
# `grafana/otel-lgtm` image as a sibling container; there is no nested Docker
# here, so its payload is copied in and started by the entrypoint. It receives
# the plugin's OTLP metrics on :4318 and serves them from Prometheus on :9090,
# which the dashboard queries.
#
# COPY --from only copies files, not the image's own ENV vars. otel-lgtm's
# run-*.sh scripts require several *_VERSION vars (set via ENV in its own
# Dockerfile) or they abort with "unbound variable", so those are captured here
# and sourced by the entrypoint before starting it.
FROM grafana/otel-lgtm:latest AS lgtm
RUN env | grep -E '_VERSION=' > /otel-lgtm.env

FROM vllm/vllm-openai-rocm:v0.28.0

LABEL org.opencontainers.image.title="AMD Agentic AI Profiling Workshop"
LABEL org.opencontainers.image.description="Profile and optimize an AI agent on AMD Instinct MI300X: Hermes Agent, MLflow telemetry, Kokoro TTS, JupyterLab."
LABEL org.opencontainers.image.vendor="AMD"
LABEL org.opencontainers.image.source="https://github.com/shailensobhee/amd-agentic-ai-profiling-workshop"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    HF_HOME=/root/.cache/huggingface \
    WORKSHOP_DIR=/workshop \
    HERMES_PROFILING_CACHE_DIR=/workshop/profiling_cache \
    HERMES_GPU=0 \
    VLLM_HERMES_PORT=8001 \
    KOKORO_PORT=8092 \
    MLFLOW_PORT=5004 \
    DASHBOARD_PORT=8501 \
    JUPYTER_PORT=8888 \
    HERMES_MODEL=meta-models/Muse-Glimmer-30B \
    HERMES_PROXY_BASE=https://notebooks.amd.com

# HERMES_PROXY_BASE is the base for the notebook's "detailed view" links and the
# URLs the entrypoint prints. It is inherited by the JupyterLab kernels the
# entrypoint launches, so the notebook picks it up automatically. Default is the
# AMD hosted-notebook proxy; override at run time for other environments, e.g.:
#   docker run -e HERMES_PROXY_BASE="" ...            # direct http://127.0.0.1:<port>/
#   docker run -e HERMES_PROXY_BASE="10.0.0.5" ...    # direct http://10.0.0.5:<port>/
#   docker run -e HERMES_PROXY_BASE="https://my-proxy" ...

# -----------------------------------------------------------------------------
# 1. OS packages
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates procps lsof espeak-ng libsndfile1 \
        ripgrep tini \
    && rm -rf /var/lib/apt/lists/*

# Grafana otel-lgtm payload (OTel collector + Prometheus + Grafana), started by
# the entrypoint to receive OTLP metrics and serve Prometheus on :9090.
COPY --from=lgtm /otel-lgtm /otel-lgtm
COPY --from=lgtm /otel-lgtm.env /otel-lgtm/lgtm.env

# -----------------------------------------------------------------------------
# 2. Kokoro TTS, dashboard, MLflow and JupyterLab dependencies
# -----------------------------------------------------------------------------
# Installed into the image's own Python so Kokoro reuses the ROCm PyTorch that
# is already present. Tested with mlflow 3.16.0.
RUN python3 -m pip install --no-cache-dir \
        kokoro soundfile fastapi 'uvicorn[standard]' \
        'streamlit>=1.30' mlflow \
        opentelemetry-sdk==1.44.0 opentelemetry-api==1.44.0 \
        opentelemetry-exporter-otlp-proto-http==1.44.0 \
        psutil requests \
        jupyterlab==4.5.7 numpy==2.4.6 'plotly>=5.18' 'pandas>=2.0' \
        matplotlib ipywidgets kaleido \
    && python3 -m pip install --no-cache-dir --upgrade "mlflow>=3.0.0" fastapi uvicorn pydantic \
    && python3 -c "import kokoro, soundfile, streamlit, mlflow, jupyterlab; print('workshop deps OK')"

# -----------------------------------------------------------------------------
# 3. Hermes Agent
# -----------------------------------------------------------------------------
RUN curl -fsSL \
        https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh \
        | bash -s -- --skip-setup --skip-browser \
    # --skip-browser omits Playwright and Chromium (about 170 MB plus a long
    # download). The workshop drives TTS and file tools, never browser tools.
    # The installer links hermes into /usr/local/bin (not ~/.local/bin), so
    # resolve it via PATH instead of hardcoding a location that can move.
    && hermes --version

# -----------------------------------------------------------------------------
# 4. hermes-otel plugin
# -----------------------------------------------------------------------------
# Install the hermes-otel plugin (per-span CPU/GPU profiling) into the image
# Python and the Hermes venv, since the plugin runs inside Hermes. Checks out the
# latest release tag. Tested on hermes-otel-v1.3.0.
RUN git clone https://github.com/briancaffey/hermes-otel.git \
        /root/.hermes/plugins/hermes_otel \
    && cd /root/.hermes/plugins/hermes_otel \
    && HERMES_OTEL_VERSION="$(git tag -l 'hermes-otel-v*' | sort -V | tail -1)" \
    && echo "Using latest hermes-otel release: ${HERMES_OTEL_VERSION:-<none, default branch>}" \
    && if [ -n "${HERMES_OTEL_VERSION}" ]; then git checkout -q "${HERMES_OTEL_VERSION}"; fi \
    && python3 -m pip install --no-cache-dir -e . \
    # Install the plugin's dependencies into the Hermes venv as well. Discover
    # the venv location rather than hardcoding it.
    && HERMES_VENV_PY="$( \
         for c in /usr/local/lib/hermes-agent/venv/bin/python \
                  /root/.hermes/hermes-agent/venv/bin/python; do \
             [ -x "$c" ] && echo "$c" && break; \
         done)" \
    && if [ -z "${HERMES_VENV_PY}" ]; then \
         echo "ERROR: could not locate the Hermes venv python" >&2; exit 1; \
       fi \
    && echo "Hermes venv python: ${HERMES_VENV_PY}" \
    && "${HERMES_VENV_PY}" -m ensurepip --upgrade \
    && "${HERMES_VENV_PY}" -m pip install --no-cache-dir -q \
        opentelemetry-api==1.44.0 opentelemetry-sdk==1.44.0 \
        opentelemetry-exporter-otlp-proto-http==1.44.0 \
        amdsmi psutil requests

# -----------------------------------------------------------------------------
# 5. Workshop content
# -----------------------------------------------------------------------------
WORKDIR /workshop
COPY tts.ipynb tts_executed.ipynb README.md ./
COPY assets/ ./assets/
COPY custom_tools/ ./custom_tools/
COPY utils/ ./utils/

# kokoro_tts_tool.py is intentionally NOT installed into Hermes here. The
# notebook's own cell installs it, matching the bare-host workflow, so the
# Step 1 (Edge TTS baseline) is not skewed by having kokoro_tts already
# registered and preferred before the notebook reaches that point.
RUN mkdir -p /workshop/outputs /root/.hermes/workspace \
             /root/.config/miopen/miopen-lockfiles

# -----------------------------------------------------------------------------
# 6. Preconfigure Hermes and the profiling plugin
# -----------------------------------------------------------------------------
# Baked in so the container starts already pointing at the local vLLM server.
RUN hermes config set model.provider custom \
    && hermes config set model.base_url "http://localhost:${VLLM_HERMES_PORT}/v1" \
    && hermes config set model.default "${HERMES_MODEL}" \
    && hermes config set compression.enabled false \
    && hermes config set model.max_tokens 16384 \
    && hermes config set terminal.cwd "${WORKSHOP_DIR}" \
    && hermes config set tool_output.max_bytes 150000 \
    && hermes config set tool_output.max_lines 5000 \
    && hermes config set tool_output.max_line_length 5000 \
    && hermes plugins enable hermes_otel --allow-tool-override

# hermes-otel plugin config, identical to the one utils/helper.sh writes: traces
# go to MLflow, CPU/GPU metrics to Grafana otel-lgtm, sampled every 100 ms.
RUN printf '%s\n' \
    'enabled: true' \
    'force_flush_on_session_end: true' \
    'capture_previews: true' \
    'capture_full_prompts: false' \
    'capture_full_responses: false' \
    'host_metrics: true' \
    'host_metrics_gpu: amd' \
    'host_metrics_interval_ms: 100' \
    'flush_interval_ms: 100' \
    'backends:' \
    '  - type: otlp' \
    '    name: mlflow' \
    "    endpoint: http://127.0.0.1:${MLFLOW_PORT}/v1/traces" \
    '    metrics: false' \
    '    logs: false' \
    '    headers:' \
    '      x-mlflow-experiment-id: "0"' \
    '  - type: otlp' \
    '    name: lgtm' \
    '    endpoint: http://127.0.0.1:4318/v1/traces' \
    '    traces: false' \
    '    metrics: true' \
    '    logs: false' \
    > /root/.hermes/plugins/hermes_otel/config.yaml

# MLflow 3.x OTLP ingestion rejects spans that carry no experiment header with
# HTTP 422, so the x-mlflow-experiment-id header above is what makes traces land.
# The plugin needs nothing from ~/.hermes/.env (it reads config.yaml above),
# matching utils/helper.sh, so no .env is written.

# -----------------------------------------------------------------------------
# 7. Entrypoint
# -----------------------------------------------------------------------------
COPY utils/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Build-time check that asserts the telemetry wiring, so a misconfigured image
# fails the build rather than at run time.
COPY utils/img-check.sh /usr/local/bin/img-check.sh
RUN chmod +x /usr/local/bin/img-check.sh && /usr/local/bin/img-check.sh

EXPOSE 8888 8501 5004 8001 8092 9090 3000

# JupyterLab is the workshop's front door; the rest come up behind it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20m --retries=3 \
    CMD curl -fsS "http://localhost:${JUPYTER_PORT}/api" >/dev/null || exit 1

# tini as PID 1 reaps orphaned/zombie descendants and forwards signals
# correctly - a plain bash script as PID 1 does neither. General Docker
# best practice for a multi-process container like this one.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve"]
