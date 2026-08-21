# =============================================================================
# AMD Agentic AI Profiling Workshop
# =============================================================================
# All-in-one workshop image for AMD Instinct MI300X.
#
# Everything runs INSIDE this single container: vLLM (Muse-Glimmer-30B), the
# Hermes agent with patched OpenTelemetry profiling, the Kokoro TTS server,
# MLflow, the Streamlit telemetry dashboard, and JupyterLab. There is no nested
# Docker, unlike utils/helper.sh which orchestrates sibling containers on a bare
# host. See utils/docker-entrypoint.sh for the in-container service manager.
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

# The ROCm vLLM nightly already provides Python 3.12, PyTorch 2.12 with HIP 7.2,
# and a working vLLM install, so the Kokoro TTS server reuses that same PyTorch
# rather than installing a second multi-gigabyte copy.
FROM rocm/device-metrics-exporter:v1.5.0 AS exporter
FROM vllm/vllm-openai-rocm:nightly

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
    HERMES_GPU=0 \
    VLLM_HERMES_PORT=8001 \
    KOKORO_PORT=8092 \
    MLFLOW_PORT=5004 \
    DASHBOARD_PORT=8501 \
    JUPYTER_PORT=8888 \
    HERMES_MODEL=meta-models/Muse-Glimmer-30B

# -----------------------------------------------------------------------------
# 1. OS packages
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl rsync ca-certificates procps lsof espeak-ng libsndfile1 \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY --from=exporter /home/amd /root/amd-dme

# -----------------------------------------------------------------------------
# 2. Muse-Glimmer support: overlay vLLM PR #51655
# -----------------------------------------------------------------------------
# Muse-Glimmer-30B is not in a released vLLM yet. PR #51655 is a Python-only
# change, so the .py files are overlaid onto the installed package with no
# kernel rebuild. This mirrors what utils/helper.sh does on a bare host, but is
# baked into the image so attendees never wait for it.
RUN git clone --filter=blob:none --no-checkout \
        https://github.com/vllm-project/vllm.git /tmp/vllm-src \
    && git -C /tmp/vllm-src fetch --depth=1 origin pull/51655/head:muse \
    && git -C /tmp/vllm-src checkout muse \
    # Resolve the INSTALLED package from outside the source tree. Running
    # python3 while cwd is /tmp/vllm-src makes Python import the local source
    # checkout instead of site-packages, which makes the overlay copy the tree
    # onto itself and leaves a half-broken vllm that aborts on import.
    && cd /root \
    && VLLM_PKG="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -1)" \
    && case "${VLLM_PKG}" in \
         /tmp/*|"") echo "Refusing to overlay onto '${VLLM_PKG}'" >&2; exit 1 ;; \
       esac \
    && echo "Overlaying PR #51655 onto ${VLLM_PKG}" \
    && rsync -a --include='*/' --include='*.py' --exclude='*' \
        /tmp/vllm-src/vllm/ "${VLLM_PKG}/" \
    && rm -rf /tmp/vllm-src \
    # Fail the BUILD (not the workshop) if the parsers did not register.
    && python3 -c "from vllm.tool_parsers import ToolParserManager; assert ToolParserManager.get_tool_parser('muse_glimmer'); print('tool parser OK')" \
    && python3 -c "from vllm.reasoning import ReasoningParserManager; assert ReasoningParserManager.get_reasoning_parser('muse_glimmer'); print('reasoning parser OK')"

# -----------------------------------------------------------------------------
# 3. Kokoro TTS, dashboard, MLflow and JupyterLab dependencies
# -----------------------------------------------------------------------------
# Installed into the image's own Python so Kokoro reuses the ROCm PyTorch that
# is already present.
RUN python3 -m pip install --no-cache-dir \
        kokoro soundfile fastapi 'uvicorn[standard]' \
        'streamlit>=1.30' mlflow==3.13.0 \
        opentelemetry-sdk==1.42.1 opentelemetry-api==1.42.1 \
        opentelemetry-exporter-otlp-proto-http==1.42.1 \
        pyrsmi==1.1.0 psutil requests cryptography \
        jupyterlab==4.5.7 'numpy>=2' 'plotly>=5.18' 'pandas>=2.0' \
        matplotlib ipywidgets \
    && python3 -c "import kokoro, soundfile, streamlit, mlflow, jupyterlab; print('workshop deps OK')"

# -----------------------------------------------------------------------------
# 4. Hermes Agent
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
# 5. hermes-otel plugin with the advanced profiling patch
# -----------------------------------------------------------------------------
# The stock plugin records which tool ran and when. The patch adds the per-span
# CPU and GPU utilization that the whole workshop is built around, so if it
# fails to apply the build must fail loudly rather than ship a dead feature.
COPY utils/hermes_advanced_profiling.patch /tmp/hermes_advanced_profiling.patch
RUN git clone https://github.com/briancaffey/hermes-otel.git \
        /root/.hermes/plugins/hermes_otel \
    && cd /root/.hermes/plugins/hermes_otel \
    && git fetch origin --tags --depth=1 \
    && git checkout hermes-otel-v0.10.0 \
    && git apply --verbose /tmp/hermes_advanced_profiling.patch \
    && rm /tmp/hermes_advanced_profiling.patch \
    && python3 -m pip install --no-cache-dir -e . \
    # The plugin executes inside Hermes' own venv, so its dependencies must be
    # installed there too. The venv location depends on where the installer put
    # Hermes, so discover it instead of hardcoding a path.
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
        opentelemetry-api==1.42.1 opentelemetry-sdk==1.42.1 \
        opentelemetry-exporter-otlp-proto-http==1.42.1 \
        pyrsmi==1.1.0 amdsmi==7.0.2 mlflow==3.13.0 \
        psutil requests cryptography

# -----------------------------------------------------------------------------
# 6. Workshop content
# -----------------------------------------------------------------------------
WORKDIR /workshop
COPY tts.ipynb tts_exec_new.ipynb README.md ./
COPY assets/ ./assets/
COPY custom_tools/ ./custom_tools/
COPY utils/ ./utils/

# The kokoro_tts tool imports `from tools.registry import ...`, a Hermes
# internal module, so it only resolves when it lives inside the Hermes tools
# package itself. Copying it to ~/.hermes/tools/ looks plausible but leaves the
# tool unimportable and the agent silently loops without ever calling it.
RUN HERMES_TOOLS_DIR="$(dirname "$(readlink -f "$(command -v hermes)")")/../tools" \
    && HERMES_TOOLS_DIR="$( [ -d /usr/local/lib/hermes-agent/tools ] \
         && echo /usr/local/lib/hermes-agent/tools \
         || echo "${HERMES_TOOLS_DIR}" )" \
    && if [ ! -d "${HERMES_TOOLS_DIR}" ]; then \
         echo "ERROR: could not locate the Hermes tools package" >&2; exit 1; \
       fi \
    && cp /workshop/custom_tools/kokoro_tts_tool.py "${HERMES_TOOLS_DIR}/" \
    && echo "Installed kokoro_tts_tool.py into ${HERMES_TOOLS_DIR}" \
    # Prove the tool actually registers, rather than trusting the file copy.
    && cd /usr/local/lib/hermes-agent \
    && venv/bin/python -c "\
from tools.registry import registry; import tools.kokoro_tts_tool; \
names = getattr(registry, 'tools', None) or getattr(registry, '_tools', {}); \
assert 'kokoro_tts' in names, f'kokoro_tts not registered, found {list(names)}'; \
print('kokoro_tts registered OK')" \
    && mkdir -p /workshop/outputs /root/.hermes/workspace \
                /root/.config/miopen/miopen-lockfiles

# -----------------------------------------------------------------------------
# 7. Preconfigure Hermes and the profiling plugin
# -----------------------------------------------------------------------------
# Baked in so the container starts already pointing at the local vLLM server.
RUN hermes config set model.provider custom \
    && hermes config set model.base_url "http://localhost:${VLLM_HERMES_PORT}/v1" \
    && hermes config set model.default "${HERMES_MODEL}" \
    && hermes config set compression.enabled false \
    && hermes config set model.max_tokens 8192  \
    && hermes config set terminal.cwd "${WORKSHOP_DIR}" \
    && hermes config set tool_output.max_bytes 150000 \
    && hermes config set tool_output.max_lines 5000 \
    && hermes config set tool_output.max_line_length 5000 \
    && hermes plugins enable hermes_otel --allow-tool-override

RUN printf '%s\n' \
    'enabled: true' \
    'force_flush_on_session_end: true' \
    'backends:' \
    '  - type: otlp' \
    "    endpoint: http://127.0.0.1:${MLFLOW_PORT}/v1/traces" \
    '    protocol: http/protobuf' \
    '    timeout: 10' \
    '    headers:' \
    '      x-mlflow-experiment-id: "0"' \
    > /root/.hermes/plugins/hermes_otel/config.yaml

# MLflow 3.x OTLP ingestion rejects spans that carry no experiment header with
# HTTP 422, and the exporter only reports that on stderr, so the sole visible
# symptom is an empty Traces tab. The header above is what makes traces land.

# Every var helper.sh appends to ~/.hermes/.env must be mirrored here. Omitting
# HERMES_TOOL_PROFILING silently disables the CPU/GPU poller: no timeline CSVs
# are written and every MLflow run ends up with zero artifacts, while still
# logging vLLM metrics, so the run looks healthy.
RUN printf '%s\n' \
    "MLFLOW_TRACKING_URI=http://127.0.0.1:${MLFLOW_PORT}" \
    "HERMES_PROFILING_OUTPUT_DIR=${WORKSHOP_DIR}/outputs" \
    "HERMES_GPU_EXPORTER_URL=http://localhost:5000/metrics" \
    "KOKORO_SERVER_URL=http://localhost:${KOKORO_PORT}" \
    'MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false' \
    'MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=1' \
    'MLFLOW_EXPERIMENT_NAME=Default' \
    'MLFLOW_RUN_NAME=Hermes_Profiling' \
    'MLFLOW_KEEP_RUN_ACTIVE=false' \
    'MLFLOW_LOGGING_LEVEL=ERROR' \
    'MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT=1' \
    'MLFLOW_DISABLE_TELEMETRY=true' \
    'HERMES_TOOL_PROFILING=1' \
    'HERMES_PROFILING_DEBUG=0' \
    'HERMES_CPU_DEBUG=0' \
    >> /root/.hermes/.env

# -----------------------------------------------------------------------------
# 8. Entrypoint
# -----------------------------------------------------------------------------
COPY utils/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Parity guard. The bare-host installer and this Dockerfile configure the same
# agent twice and drift silently, so assert the telemetry wiring at build time
# rather than discovering a dead dashboard on a live GPU host.
COPY utils/img-check.sh /usr/local/bin/img-check.sh
RUN chmod +x /usr/local/bin/img-check.sh && /usr/local/bin/img-check.sh

EXPOSE 8888 8501 5004 8001 8092

# JupyterLab is the workshop's front door; the rest come up behind it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20m --retries=3 \
    CMD curl -fsS "http://localhost:${JUPYTER_PORT}/api" >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve"]
