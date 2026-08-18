# Running the workshop in Docker

This image packages the entire workshop so attendees need one command instead of
a manual backend setup. It targets **AMD Instinct MI300X** with ROCm.

Image: `shailensobhee1/amd-agentic-ai-profiling:mi300x`

---

## What is inside

Everything runs as a local process inside the single container, so there is no
nested Docker and no host setup beyond the GPU devices.

| Component | Port | Notes |
| :--- | :--- | :--- |
| JupyterLab | `8888` | The workshop front door. Open `tts.ipynb`. |
| Telemetry dashboard (Streamlit) | `8501` | Spans, CPU/GPU timeline, tool breakdown. |
| MLflow tracking server | `5004` | Traces and hardware metrics. |
| vLLM (Muse-Glimmer-30B) | `8001` | The agent's model, OpenAI-compatible API. |
| Kokoro TTS server | `8092` | Local TTS on the MI300X. |

Also baked in: the Hermes Agent (preconfigured to use the local vLLM), the
`hermes-otel` plugin with the advanced profiling patch applied, the custom
`kokoro_tts` tool, the notebooks, and all workshop assets.

> **How this differs from `utils/helper.sh`.** On a bare host, `helper.sh`
> launches vLLM and the metrics exporter as *sibling Docker containers*. That
> cannot work unchanged inside an image without mounting the host Docker socket,
> so the container uses `utils/docker-entrypoint.sh` instead, which starts the
> same services as local processes and reaches the GPU directly. Both paths
> produce the same workshop.

---

## Prerequisites

- An AMD Instinct MI300X (or another ROCm-supported AMD Instinct GPU)
- A working ROCm driver on the host, so `/dev/kfd` and `/dev/dri` exist
- Docker
- About 60 GB of free disk for the model weights, plus room for the image

---

## Quick start

```bash
docker run -d --name amd-agentic-ai-profiling \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  --ipc=host --shm-size 16G \
  -p 8888:8888 -p 8501:8501 -p 5004:5004 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  shailensobhee1/amd-agentic-ai-profiling:mi300x
```

Then watch it come up:

```bash
docker logs -f amd-agentic-ai-profiling
```

When the log prints `All services are ready`, open
`http://<host>:8888/lab/tree/tts.ipynb`.

> **First start downloads about 60 GB of model weights.** Mounting the Hugging
> Face cache as shown means the download is paid once per host, not once per
> container. Later starts come up in minutes.

### Why each flag is needed

| Flag | Why |
| :--- | :--- |
| `--device=/dev/kfd --device=/dev/dri` | GPU access. Without these the entrypoint stops immediately with a clear error. |
| `--security-opt seccomp=unconfined` | Required by ROCm. |
| `--group-add video` | Grants access to the render nodes. |
| `--ipc=host --shm-size 16G` | Shared memory for PyTorch and vLLM workers. |
| `-v ...huggingface...` | Avoids re-downloading the model on every run. |

---

## Useful variations

**Protect JupyterLab with a token** (it is open by default for workshop convenience):

```bash
-e JUPYTER_TOKEN=choose-a-token
```

**Pick a different GPU:**

```bash
-e HERMES_GPU=3
```

**Share a busy GPU.** vLLM sizes its allocation against *free* VRAM, so it fails
to start if another process already holds a large share. The container defaults
to `0.80` instead of vLLM's `0.92` for that reason, and prints how much VRAM is
already in use before starting. Lower it further if needed:

```bash
-e GPU_MEMORY_UTILIZATION=0.70
```

**Open a shell instead of starting the services:**

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  shailensobhee1/amd-agentic-ai-profiling:mi300x bash
```

**Persist notebook edits and outputs to the host:**

```bash
-v "$PWD/work:/workshop/outputs"
```

---

## Building it yourself

From the repository root:

```bash
docker build -f utils/Dockerfile -t amd-agentic-ai-profiling:mi300x .
```

The build asserts that the Muse-Glimmer tool and reasoning parsers register
after the vLLM PR #51655 overlay, and that the profiling patch applies, so a
broken image fails at build time rather than during the workshop.

---

## Verified on hardware

This image was built and tested on a real AMD Instinct MI300X, not just
smoke-checked. The following were confirmed end to end inside the container:

| Check | Result |
| :--- | :--- |
| All five services reach a healthy endpoint | MLflow 5s, Kokoro 45s, dashboard 5s, JupyterLab 5s, vLLM about 4 min from a warm cache |
| vLLM serves the model | `max_model_len = 131072`, above the 64K minimum Hermes requires |
| Real GPU inference | Chat completion returned the expected text with a populated `reasoning` field, confirming the `muse_glimmer` parsers |
| Kokoro TTS on the GPU | `POST /tts/wav` returned 200 and produced 94 s of 24 kHz audio |
| Agent tool use | Hermes called `kokoro_tts` with the `text_file` parameter and wrote real WAV output |
| Profiling telemetry | MLflow recorded runs carrying vLLM cache-hit metrics |

---

## Troubleshooting

| Symptom | Cause and fix |
| :--- | :--- |
| `/dev/kfd is missing` on start | The GPU flags were omitted. Use the full `docker run` above. |
| Stuck at `Waiting for vLLM` | Normal on first start while weights download. Follow progress with `docker exec amd-agentic-ai-profiling tail -f /workshop/logs/vllm.log`. |
| A service exited and the container stopped | The entrypoint prints the tail of every log on failure. Inspect with `docker logs amd-agentic-ai-profiling`. |
| Dashboard is empty | Run a notebook cell first, then click **Fetch** in the dashboard and select the newest run. |
| Port already in use | Change the host side of the mapping, for example `-p 9999:8888`. |
| `vLLM could not reserve enough VRAM` | Another process is holding GPU memory. The entrypoint reports how much at startup. Retry with a lower `-e GPU_MEMORY_UTILIZATION=0.70`. |
| Agent loops without producing audio | The `kokoro_tts` tool failed to register. The image asserts this at build time, so this should not happen. Check with `docker exec <name> ls /usr/local/lib/hermes-agent/tools/kokoro_tts_tool.py`. |

Logs for every service live in `/workshop/logs/` inside the container.
