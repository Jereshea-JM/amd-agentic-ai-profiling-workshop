#!/usr/bin/env python3
"""
Regenerate tts.ipynb for the AMD Agentic AI Profiling Workshop.

The notebook is generated from this script so it stays reproducible and diffable:
  * Every cell's source is a literal in this file, so the script is
    self-contained: it reads no other notebook.
  * The matplotlib chart cell is rewritten here with AMD branding.
  * Images are embedded as inline base64 data URIs, each with alt text and a caption.

Run:  python scripts/build_tts_notebook.py
"""
import base64
import json
import os
import sys

# --standalone builds a single portable notebook (tts_standalone.ipynb): it adds
# an early "Notebook preparation" cell that fetches the few support files the
# cells need (utils/, custom_tools/, the dashboard logo) from the public repo,
# and it bakes every remaining external image inline as base64 so the one .ipynb
# ships on its own. Without the flag we build the repo-native tts.ipynb, whose
# support files and assets/ already sit beside it in the checkout.
STANDALONE = "--standalone" in sys.argv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
NB_PATH = os.path.join(ROOT, "tts_standalone.ipynb" if STANDALONE else "tts.ipynb")
DIAGRAMS = os.path.join(ROOT, "assets", "diagrams")
OUTPUTS = os.path.join(ROOT, "assets", "outputs")

# Public repo the standalone fetch cell pulls its support files from.
STANDALONE_REPO = "https://github.com/shailensobhee/amd-agentic-ai-profiling-workshop.git"
STANDALONE_BRANCH = "main"

# ---- cell builders ----------------------------------------------------------
_cells = []


def md(src):
    _cells.append({"cell_type": "markdown", "metadata": {}, "source": _split(src)})


def code(src):
    _cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": src if isinstance(src, list) else _split(src),
    })


def _split(s):
    """Store source as a list of lines with trailing newlines, like nbformat."""
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def img(name, alt, caption, width="88%", subdir="diagrams"):
    """Embed a PNG as an inline base64 data URI with alt text + caption."""
    path = os.path.join(ROOT, "assets", subdir, name)
    data = _b64(path)
    html = (f'<p align="center">\n'
            f'<img alt="{alt}" '
            f'src="data:image/png;base64,{data}" width="{width}">\n</p>\n\n'
            f'<p align="center"><sub><i>{caption}</i></sub></p>')
    md(html)


def _img_data_uri(relpath):
    """Return a base64 data URI for an asset given a repo-relative path."""
    return "data:image/png;base64," + _b64(os.path.join(ROOT, relpath))


# Overview cells shown after each profiling run. The Step 2 cell exposes the
# link-base dropdown and sets HERMES_PROXY_BASE; the kokoro cells reuse it.
_OVERVIEW = '''import importlib, sys, os, logging, warnings

sys.path.insert(0, os.path.abspath("utils"))

logging.disable(logging.WARNING)   # mute Streamlit's "missing ScriptRunContext" etc.
warnings.filterwarnings("ignore")  # mute plotly/mlflow UserWarning/FutureWarning etc.

import hermes_profiler
importlib.reload(hermes_profiler)          # pick up edits without a kernel restart

SESSION_ID = None   # a run ID, or None to auto-fetch the latest session
hermes_profiler.show_session_overview(SESSION_ID)

logging.disable(logging.NOTSET)'''



# =============================================================================
# NOTEBOOK CONTENT
# =============================================================================

# ---- 0. Title / intro -------------------------------------------------------
md(
"""<p align="center">
<img alt="AMD" src="data:image/png;base64,"""
+ _b64(os.path.join(ROOT, "assets", "images", "amd_logo.png")) +
"""" width="150">
</p>

<h1 align="center">Profiling &amp; Optimizing an AI Agent on AMD Instinct&trade; GPUs</h1>

<p align="center">
<b>An observability-driven optimization workshop</b><br>
<sub>Hermes Agent &middot; MLflow &amp; Grafana otel-lgtm telemetry &middot; AMD Instinct&trade; MI300X &middot; ROCm&trade;</sub>
</p>
"""
+ ("""
**Author**: Shailen Sobhee, Jereshea John Mary, Sabira Shaik  
**Knowledge level**: Intermediate
""" if STANDALONE else "") +
"""
---

## Welcome

This is a hands-on, **beginner-friendly** tutorial. You will run a real AI agent,
watch where it spends its time, find the one slow step, optimize it, and prove the
speed-up with hardware telemetry. No prior profiling experience is assumed.

[**Hermes Agent**](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)
is an open-source, autonomous AI agent framework by Nous Research. Unlike a plain
chatbot, an *agent* can **plan, choose tools, and complete multi-step tasks on its
own**, deciding which tools to call and in what order to reach a goal.

> **Two words to know first**
>
> | Term | Meaning |
> | :--- | :--- |
> | **Tool** | A single action the agent can perform, exposed as a callable function (for example, "convert this text to speech" or "read a file"). At each step the agent picks the tool that fits. |
> | **Skill** | A higher-level, reusable ability: a packaged set of instructions that may combine several tools to handle a larger task. |
>
> Hermes ships with built-in tools, and you can **add your own**. The Kokoro
> text-to-speech tool used here is exactly such a custom tool.

## Why profiling matters

When an agent runs, some steps are fast and some are slow. **Profiling** is how we
measure each step so we can see exactly where the time goes.

An agent is only as fast as the tools it calls. A single slow tool can dominate an
entire run even when the model itself is quick. Finding and fixing that one tool is
often the difference between a sluggish agent and a responsive one.

## What you will learn

By the end of this session you will be able to:

- Run a Hermes agent and capture its telemetry
- Read the telemetry dashboard to spot a slow tool
- Swap in an optimized version of that tool
- Compare before and after, and see how hardware usage changes
"""
)

img("01_pipeline.png",
    "Pipeline: an input text file flows into the Hermes Agent, which calls a "
    "TTS tool, which produces audio output. The workshop focus is profiling "
    "and optimization.",
    "Our example workflow. Text-to-speech is only the example; the real subject "
    "is profiling and optimization.",
    width="92%")

md(
"""> **Keep this in mind throughout.** The focus is **profiling and optimization**.
> Text-to-speech is simply the example we use to expose a bottleneck and fix it.
"""
)

# ---- 1. Prerequisites -------------------------------------------------------
md(
"""---

## Prerequisites

This tutorial was developed and tested with the setup below.

### Operating system
**Ubuntu 24.04.** Ensure your system is running Ubuntu 24.04.

### Hardware
**AMD Instinct&trade; MI300X GPU (192 GB VRAM).** This tutorial was tested on a
single MI300X, which comfortably hosts both the Muse-Glimmer-30B model and the
Kokoro TTS model at once. Use an AMD Instinct&trade; GPU with ROCm support that meets the
official requirements.

### Software

**ROCm&trade; 7.2.** Install and verify ROCm using the ROCm install guide, then
confirm your GPUs are visible:

```bash
amd-smi
```

> **Note.** For ROCm 6.4 and earlier, use `rocm-smi` instead.

**vLLM ROCm image.** The agent's model is served with vLLM, using AMD's prebuilt
`vllm/vllm-openai-rocm:v0.28.0` release image, which includes Muse-Glimmer-30B
support. AMD also provides other prebuilt ROCm images (PyTorch, Ubuntu 22.04 /
24.04) you can reuse for ROCm work.

**Python 3.12** (with `venv` and `pip`) runs the Kokoro server, MLflow, and this
notebook.
"""
)

# ---- 3. What utils/helper.sh does -------------------------------------------------
md(
"""## What `utils/helper.sh` sets up

The backend is already running: `utils/helper.sh` (which you start per the README, or
which the container image runs automatically) handles the whole setup. It installs the
dependencies, starts the services below, and configures the `hermes-otel` plugin for
fine-grained profiling (a 100 ms CPU/GPU sampling interval, much finer than the
plugin's default, so the timelines can resolve per-tool activity).
"""
)

img("02_architecture.png",
    "Architecture of the backend that utils/helper.sh starts: the Hermes Agent runtime "
    "(vLLM, Muse-Glimmer-30B) calls the Kokoro TTS server on the MI300X; "
    "hermes-otel sends execution traces to the MLflow tracking server and CPU/GPU "
    "metrics to Grafana otel-lgtm; the Streamlit telemetry dashboard reads traces "
    "from MLflow and metrics from otel-lgtm to show one clear view.",
    "One command brings up the whole observability stack.",
    width="94%")

md(
"""Behind the scenes it brings up:

| Service | Role |
| :--- | :--- |
| **Hermes backend** (vLLM &middot; Muse-Glimmer-30B) | The agent's "brain": the model that plans and picks tools. |
| **Hermes OTel** | The plugin that instruments the agent. `utils/helper.sh` writes its config file with two OpenTelemetry backends: it sends execution **traces** (spans, timings, tokens) to the MLflow tracking server, and hardware **metrics** to Grafana `otel-lgtm`. `utils/helper.sh` sets it to sample `psutil` (CPU) and `amdsmi` (GPU) every 100 ms (much finer than the plugin's default), so the timelines have the resolution to see per-tool activity. |
| **MLflow tracking server** | Stores the execution **traces** the dashboard visualizes. |
| **Grafana `otel-lgtm`** | Receives the CPU/GPU **metrics** over OTLP and stores them (Prometheus), which the dashboard queries for the utilization timelines: system-wide GPU%, the Hermes process (plus children) CPU%, and per-tool CPU/GPU%. |
| **Telemetry dashboard** | A custom Streamlit page that reads traces from MLflow and metrics from `otel-lgtm` to give one clear view of each run. |
| **Kokoro TTS server** | The local TTS engine used here as a faster, self-hosted alternative to the default cloud (Edge) TTS provider, avoiding the network round-trip and per-request cost. |

> **About the model.** **Muse-Glimmer-30B** is a dense vision-language model built
> for agentic work: a 52-layer text decoder (hidden size 6656) plus a ~1.8B
> ViT-G/14 perception encoder, 128K trained context, BF16. Apache 2.0, knowledge
> cutoff January 4 2026, trained on 100+ languages.

The cells further down in this notebook visualize this telemetry directly, reading
the traces from MLflow and the CPU/GPU metrics from `otel-lgtm`.
"""
)

# ---- 4. Check your setup ----------------------------------------------------
md(
"""## Check your setup

Before sending the agent any work, confirm the `hermes` command is available in
this notebook's environment. The cell below adds the usual install locations to
`PATH`, prints the path to the Hermes binary, then `Hermes is ready.`

> **Why the `PATH` line?** Depending on how Hermes was installed, the binary
> lands in either `/usr/local/bin` (the container and root installs used by
> `utils/helper.sh`) or `~/.local/bin` (a per-user pip install). A JupyterLab
> kernel does not always inherit the login shell's `PATH`, so without this cell
> `!which hermes` can find nothing even though Hermes is correctly installed.

> **If nothing prints,** the notebook still cannot find Hermes. Make sure
> `utils/helper.sh` has finished starting up, and that the notebook was launched
> from the same environment.
"""
)
# This cell adds a PATH fix-up above the `which` call because a JupyterLab kernel
# does not always inherit the login shell's PATH. Both /usr/local/bin (container /
# root installs) and ~/.local/bin (per-user pip installs) are added explicitly,
# since `hermes` can resolve from either depending on how it was installed.
code(
"""import os

# Hermes may live in either location depending on how it was installed:
#   /usr/local/bin  -> container / root install performed by utils/helper.sh
#   ~/.local/bin    -> per-user pip install
# A JupyterLab kernel does not always inherit the login shell's PATH, so add
# both and let `which` report the one actually in use.
for _p in ("/usr/local/bin", os.path.expanduser("~/.local/bin")):
    if _p not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + _p

!which hermes && echo "Hermes is ready."
"""
)

# ---- 4b. Standalone: fetch the support files this notebook needs ------------
# In the repo checkout these live beside the notebook (utils/, custom_tools/,
# assets/). The standalone notebook ships alone, so it pulls just those few
# files from the public repo into the working directory, mirroring the pattern
# used by the Google ADK tutorial in this collection. Idempotent: re-running
# refreshes the checkout without failing if it already exists.
if STANDALONE:
    md(
"""---

## Notebook preparation

This is a **standalone** notebook: everything it needs travels with it, except a
few small support files that back specific cells (the telemetry dashboard module
`utils/hermes_profiler.py` and its Streamlit theme, the custom `kokoro_tts` tool,
and the dashboard logo). The cell below fetches just those files from the public
workshop repo into your working directory, so the rest of the notebook runs
exactly as it would inside the repo. All diagrams and screenshots are already
embedded in the notebook itself.

**What the cell below does, step by step:**

* **Clones only what is needed.** It does a shallow, sparse checkout of the public
  workshop repo (`--depth 1 --filter=blob:none --sparse`), so it pulls three small
  folders instead of the whole repository with its large executed notebooks.
* **Fetches three things:** `utils/hermes_profiler.py` (the telemetry dashboard
  module) plus its Streamlit theme, `custom_tools/kokoro_tts_tool.py` (the custom
  agent tool this notebook deploys into Hermes), and `assets/images/amd_logo.png`
  (the dashboard logo).
* **Puts them where the cells expect them.** It copies the files into your working
  directory, removes the temporary checkout, and prints an `[OK]` line listing each
  file so you can confirm setup succeeded.

Nothing here downloads a model or a picture: every diagram and screenshot is baked
into the notebook, so this cell fetches only the small pieces of runnable code the
later cells import.

Run it once at the start. It is safe to re-run: it refreshes the files in place.
"""
    )
    code(
'''%%bash
set -euo pipefail

REPO_URL="{repo}"
BRANCH="{branch}"
CLONE_DIR=".nb_support_checkout"

# Fetch only what the later cells import or deploy, not the whole repo. A shallow,
# sparse checkout keeps this fast and avoids pulling the large executed notebooks.
rm -rf "$CLONE_DIR"
git clone --quiet --depth 1 --branch "$BRANCH" --filter=blob:none --sparse "$REPO_URL" "$CLONE_DIR"
( cd "$CLONE_DIR" && git sparse-checkout set utils custom_tools assets/images >/dev/null )

# Place the files where the notebook cells expect them (working directory root).
mkdir -p utils custom_tools assets/images
cp -r "$CLONE_DIR/utils/." utils/
cp -r "$CLONE_DIR/custom_tools/." custom_tools/
cp -r "$CLONE_DIR/assets/images/." assets/images/
rm -rf "$CLONE_DIR"

echo "[OK] Support files ready:"
echo "     utils/hermes_profiler.py        (telemetry dashboard module)"
echo "     utils/.streamlit/config.toml    (dashboard theme)"
echo "     custom_tools/kokoro_tts_tool.py (custom agent tool)"
echo "     assets/images/amd_logo.png      (dashboard logo)"
'''.format(repo=STANDALONE_REPO, branch=STANDALONE_BRANCH)
    )

# ---- 5. Prepare input text --------------------------------------------------
md(
"""---

## Prepare your input text

Text-to-speech is the example use case we profile, so first we need a passage to
synthesize. In the cell below we let Hermes itself write the input passage and save
it to `input_text.txt`, which is then passed to the TTS tool.
"""
)
# The input-passage prompt (~8,450 characters). A longer passage makes the
# batching improvement later in the notebook more visible.
code('''!hermes chat --yolo --oneshot -q "Write ONE single continuous paragraph of nearly about 1,000 words (roughly 8000 characters) on the topic of AMD GPUs. It must be a single block of flowing prose: do NOT number the sentences, do NOT put each sentence on its own line, and do NOT use any line breaks, headings, bullet points, lists, quotes, code, or special symbols. Use normal sentence punctuation (periods and commas) so it reads naturally for text-to-speech. Write the whole passage as one block with no newline characters. Save it to 'input_text.txt' in the current directory, overwriting existing content, using your write/file tool. Do not read any other file."''')

md(
"""> **Make it your own.** Feel free to pick a different topic for the passage, and
> to vary its length. Longer input makes the batching win later in the notebook
> much more visible.
"""
)

# ---- 6. Step 1: Baseline ----------------------------------------------------
md(
"""---

## Step 1 &middot; Baseline: Edge TTS

Every profiling exercise needs a starting point. Ours is **Edge TTS**, the
text-to-speech provider Hermes uses by default.

The command below asks the agent to read the generated `input_text.txt` and speak
it.
"""
)
# Edge TTS baseline. The terser prompt ("convert the text and save the audio")
# is deliberate: leaving the agent to work out how is what produces the repeated
# tool calls the "Edge TTS observations" section below discusses, which is the
# behaviour this step exists to demonstrate.
code('''!hermes chat --yolo --oneshot -q "Convert the entire text in input_text.txt to audio and save it in output_audio.mp3"''')

# ---- 7. Step 2: Profiling ---------------------------------------------------
md(
"""---

## Step 2 &middot; Profiling

**Watch the Hermes output cell** and you will notice logs similar to this:

```text
[hermes-otel] ✓ mlflow at http://127.0.0.1:5004/v1/traces (traces only)
[hermes-otel] ✓ lgtm at http://127.0.0.1:4318/v1/traces (query only)
[hermes-otel] ✓ Live dashboard store active
[hermes-otel] ✓ Host metrics sampler on (every 100 ms, gpu=amd)
[hermes-otel] Registered 13 hooks
```

This appears because `utils/helper.sh` configures the hermes-otel plugin to send this
session's execution **traces** to the MLflow server (its CPU/GPU **metrics** go to
Grafana `otel-lgtm` separately). The telemetry dashboard in the next section reads
both: traces from MLflow and metrics from `otel-lgtm`.

The three steps below open the raw MLflow interface. Skip them if you only want the
high-level overview the dashboard gives you.

1. To browse the detailed log, open your local MLflow interface (typically at
   `http://<system_ip>:5004`).
2. Open the **Traces** tab in the left panel and find your recent request.
3. In parallel, open the **Evaluation Runs** tab and find your hermes-session-id
   request.
"""
)

md('''### Launching the profiling dashboard

To make the MLflow data easier to read, we built a custom Streamlit dashboard on
port `8501`, started for you by `utils/helper.sh`. The cell below resolves your
server address and gives you a direct link.

> **Which link should you click?** Use the **`localhost`** link when the browser
> runs on the same machine as the workshop (or when you forwarded the port with
> `ssh -L 8501:localhost:8501`). Use the **server-IP** link when you are hitting a
> remote machine directly and port `8501` is reachable from your network.

**The dashboard has five tabs:**

1. **Overview:** Plots the span waterfall for the agent's flow alongside CPU and
   GPU utilization at each span. A toggle switches between the standalone CPU/GPU
   timeline (the default) and the full-session waterfall correlated with it. A
   tool-breakdown table sits below the chart.
2. **CPU / GPU separate:** Shows the CPU and GPU graphs individually, with an
   option to view the raw `.csv` files the Overview charts are plotted from.
3. **Context & tools:** Charts how the agent's context grows step by step across the session, plus a per-turn breakdown and every tool    outcome, including failures.
4. **Traces:** Provides a direct MLflow link for each turn in the session.
5. **Analysis:** Feeds the MLflow traces plus each tool's execution time to the
   local `hermes` CLI and reports how the agent could be improved. Depending on
   the length of the traces this can take around five minutes.
''')
# This cell prints both the localhost link (for the SSH-port-forward and
# container paths, where only localhost resolves) and the server-IP link (for an
# attendee hitting a remote box directly), and labels which is which.
code('''# NOTE: Skip this cell if you are using the default AMD hosted-notebook proxy.
# If you are running locally or using a custom proxy, uncomment and set the appropriate base below:
# os.environ["HERMES_PROXY_BASE"] = ""  # Use "" for local 127.0.0.1 links, or "https://your-custom-proxy"''')
code(_OVERVIEW)

# ---- 8. Step 3: Analyzing the logs -----------------------------------------
md(
"""---

## Step 3 &middot; Analyzing the run

After each Hermes execution, review the telemetry to understand where time was
spent and which tool drove the latency. Here is how the pieces fit together:

- Every Hermes run is recorded as a **session** with a unique session id: its
  execution traces are logged to MLflow and its CPU/GPU metrics to `otel-lgtm`.
- In the dashboard, click **Fetch** to load the recorded runs. The **most recent
  run appears at the top**, followed by older ones.
- **Select** the run you want (usually the latest), then click **Load / Reload**.
  The dashboard draws a picture of what happened, with no need to copy or type a
  session id.

> **The loop is always the same:** run the agent, click **Fetch**, select the
> latest run, click **Load / Reload**, and inspect.

Once the run loads, explore what the dashboard shows for this execution:

- The timeline of the run
- When the TTS tool ran, and for how long
- Hardware metrics over time (GPU and CPU utilization)
- Overall execution latency
"""
)

md('''<div align="center">

![The AMD Agent Telemetry dashboard Overview tab for a Hermes session. A span waterfall shows the agent, LLM and API spans plus the text_to_speech tool span highlighted in orange as the longest at 24.44 seconds, and a CPU/GPU utilization time series below tracks hardware use across the run.](%s)

<sub>*The Overview tab of the telemetry dashboard. The orange text_to_speech span is the longest single step, and the utilization chart shows the GPU is mostly idle while it runs. That gap is exactly the bottleneck we will fix.*</sub>

</div>''' % (
    _img_data_uri("assets/images/dashboard/dashboard_overview.png") if STANDALONE
    else "./assets/images/dashboard/dashboard_overview.png"))

md(
"""<details>
<summary><b>How this works under the hood</b></summary>

<br>

The dashboard builds this view from two sources: the session's spans come from the
MLflow traces, and the CPU/GPU timeline is queried from the metrics stored in
`otel-lgtm` (Prometheus). It compiles both into one human-readable summary.

</details>
"""
)

# ---- 9. Edge TTS observations ----------------------------------------------
md(
"""### Edge TTS observations

In the telemetry dashboard, find the Hermes session id for the Edge TTS run you
just did, then look at the audio Edge produced and note a few things:

- **No local setup.** Edge is cloud-based. Hermes sends text to the service and
  receives audio back, which makes it a convenient baseline: nothing to install,
  nothing to configure.
- **Input-length limit.** Edge TTS has a **5,000-character input limit**. The
  Hermes wrapper handles longer inputs by splitting them into chunks (a
  10K-character input becomes two chunks, and so on, depending on the chunking
  logic).
- **Audio stitching.** The chunks are normally combined into one output, but for
  very large inputs they may not stitch correctly, producing several audio files
  for a single request.
- **Short inputs.** For shorter inputs, Edge TTS works well and is convenient.
- **Privacy.** Because Edge is cloud-based, the text leaves the local machine for
  processing. For privacy-sensitive workloads, a local model is preferable.

> **Why this step can produce messy, repeated tool calls.** The built-in
> `text_to_speech` tool accepts text only *inline* (there is no file-path
> parameter). For a long passage the agent cannot pass the whole text in one call,
> so it often improvises with extra steps (chunked file reads, `wc`/`head`/`cat`,
> temporary copies, small Python snippets) while working around the limit.

The length limit, the stitching edge cases, the privacy consideration and the
network round-trip on every request are together our reason to move to a local
model next.
"""
)

# ---- 10. Step 4: Local Kokoro ----------------------------------------------
md(
"""---

## Step 4 &middot; Local TTS with Kokoro

**Kokoro** is a text-to-speech model that runs entirely on the local machine. Here
it is served by the Kokoro TTS server that `utils/helper.sh` started for you on an **AMD
Instinct&trade; MI300X GPU**.

The server is a lightweight FastAPI + Uvicorn wrapper around the Kokoro model. The
wrapper keeps the model resident in GPU memory between requests, so each synthesis
call is fast instead of paying model-load overhead every time.

Because inference happens locally, your text never leaves the system, a natural fit
for privacy-sensitive workloads. It also means run speed now depends on how well the
tool uses the local hardware, which is exactly what we want to profile.
"""
)

md(
"""### The custom tool: `kokoro_tts`

`kokoro_tts` is a custom tool we added to Hermes. It sends text to the local Kokoro
server and returns spoken audio as a **WAV** file (uncompressed, so it preserves the
original quality). Extending Hermes with your own tools and skills is one of its
strengths, and `kokoro_tts` is exactly that: a local text-to-speech tool.

The cell below installs the tool into Hermes. It must live **inside the Hermes
`tools` package**, because the file does `from tools.registry import ...`. Copying
it anywhere else (`~/.hermes/tools/`, for example) looks plausible but leaves the
tool unimportable, and the agent then loops without ever calling it, with no error
message to explain why. The cell locates the real package and verifies the tool
registers, rather than assuming the copy worked.
"""
)
# A %%bash cell that copies the custom tool into the Hermes tools package. It
# resolves the package for both layouts - a per-user install
# ($HOME/.hermes/hermes-agent/tools) and the container image
# (/usr/local/lib/hermes-agent/tools) - and asserts the tool registers rather
# than trusting a bare `cp`.
code(
'''%%bash
set -uo pipefail

SRC="./custom_tools/kokoro_tts_tool.py"
if [ ! -f "$SRC" ]; then
    echo "[ERROR] $SRC not found. Run this from the repository root."
    exit 1
fi

# Resolve the Hermes installation, whichever layout this machine uses:
# a per-user install (~/.hermes/hermes-agent) or a system one (/usr/local).
# Require both tools/ and venv/bin/python, since the verification step below
# runs that interpreter.
HERMES_ROOT=""
for cand in "$HOME/.hermes/hermes-agent" /usr/local/lib/hermes-agent; do
    if [ -x "$cand/venv/bin/python" ] && [ -d "$cand/tools" ]; then
        HERMES_ROOT="$cand"
        break
    fi
done

if [ -z "$HERMES_ROOT" ]; then
    echo "[ERROR] Could not find a complete Hermes install (tools/ + venv/)."
    echo "        Has hermes finished installing?"
    exit 1
fi

echo "[INFO] Deploying custom Kokoro TTS tool to $HERMES_ROOT/tools ..."
cp "$SRC" "$HERMES_ROOT/tools/"
echo "[OK] Copied kokoro_tts_tool.py -> $HERMES_ROOT/tools/"

# Verify the tool registers, not just that the file copied.
"$HERMES_ROOT/venv/bin/python" -c "
import sys; sys.path.insert(0, '$HERMES_ROOT')
from tools.registry import registry
import tools.kokoro_tts_tool
names = getattr(registry, 'tools', None) or getattr(registry, '_tools', {})
assert 'kokoro_tts' in names, f'kokoro_tts NOT registered; found {sorted(names)}'
print('[OK] kokoro_tts is registered and callable by the agent.')
"
'''
)

md(
"""**Parameters of the `kokoro_tts` tool**

| Parameter | Purpose |
| :--- | :--- |
| `text` | The full text to synthesize, in one call. |
| `text_file` | Path to a UTF-8 file to synthesize. Preferred for long text, so the agent passes a path instead of inlining the whole passage. |
| `mode` | The inference strategy, `sequential` or `batched`. `sequential` is the native Kokoro baseline; `batched` is the optimization we add in Step 5. This parameter is what lets us compare the two. Defaults to `sequential`. |
| `voice` | Voice name (default `af_heart`). |
| `batch_size` | Sentences per GPU forward pass when `mode` is `batched` (default `16`). Ignored in `sequential` mode. |
| `output_path` | Where to save the WAV (defaults to `~/.hermes/audio_cache/`). |

The tool is defined in `custom_tools/kokoro_tts_tool.py` and backed by
`utils/kokoro_server.py`. Let's run it on our input and profile how it performs.
"""
)
code('''!hermes chat --yolo --oneshot -q "Use the kokoro_tts tool and pass the file path 'input_text.txt' as the text_file parameter to convert the text to speech. Do not read the file yourself"''')
code(_OVERVIEW)

md(
"""### How `kokoro_tts` works, and why the first run is slow

`kokoro_tts` runs the Kokoro TTS model locally on the AMD MI300X GPU. In its default
mode it processes the input **one sentence at a time** and saves the result as a WAV
file.

Load this run in the dashboard the same way as before: click **Fetch**, select
the run at the top of the list, then click **Load / Reload**. You should see:

- The **`kokoro_tts`** span occupies the largest part of the timeline, so it is the
  primary contributor to end-to-end latency.
- **GPU utilization stays low for most of the run.** The MI300X is not being fully
  used.
- Because sentences are processed one at a time, each inference gives the GPU only a
  small amount of work, which is inefficient.
- As the number of sentences grows, the tool runs more inference passes, so latency
  grows **almost linearly**.

> The tool works and produces correct audio, but it clearly under-uses the GPU.
> That is the bottleneck the dashboard reveals, and the motivation to optimize the
> tool next.
"""
)

# ---- 11. Step 5: Optimize with batching ------------------------------------
md(
"""---

## Step 5 &middot; Optimize the tool with batching

The dashboard showed the bottleneck: the tool feeds the GPU **one sentence at a
time**. So we edited `kokoro_tts` to add an optimized **`batched`** mode.

Instead of processing text one piece at a time (the original behavior, which we now
call **`sequential`** mode), the new **`batched`** mode groups many sentences and
sends them to the GPU in a single forward pass, giving the hardware much more work to
do at once.
"""
)

img("05_batching.png",
    "Comparison of sequential and batched processing. In sequential mode five "
    "sentences s1 to s5 each take their own forward pass, causing five GPU "
    "launches and low utilization. In batched mode the same five sentences are "
    "grouped and length-bucketed into a single GPU launch with high utilization.",
    "Sequential mode issues one GPU launch per sentence; batched mode groups them "
    "into a single launch, keeping the GPU busy.",
    width="94%")

md(
"""We re-run the exact same input, this time asking for `mode='batched'`, and
compare.
"""
)
code('''!hermes chat --yolo --oneshot -q "Use the kokoro_tts tool with the mode parameter set to 'batched', and pass the file path 'input_text.txt' as the text_file parameter. Do not read the file yourself"''')
code(_OVERVIEW)

md(
"""Load this run in the dashboard the same way as before: click **Fetch**, select
this run (it appears at the top), then click **Load / Reload**. Then compare it
side by side with the sequential run.

### What changed under the hood

`mode='batched'` uses our optimized implementation. Kokoro does not support native
batching, so we modified the inference pipeline to process multiple sentences in a
single GPU forward pass instead of one at a time:

- **Batching.** Multiple sentences are grouped so the GPU processes more work per
  forward pass, improving hardware utilization.
- **Length bucketing.** Sentences of similar length are grouped into the same batch,
  reducing wasted padding.
- **Correct batched processing.** Padding and attention masks keep each sentence
  independent, and the final audio is trimmed back to its true length. See
  `utils/kokoro_server.py` for the source code.

### What you should see in the dashboard

- With `mode='batched'`, `kokoro_tts` completes **significantly faster** than
  sequential.
- The GPU does **more work per forward pass**, so utilization is higher and overhead
  is lower.
- Fewer GPU launches overall, because sentences are processed together rather than
  one at a time.
- CPU activity stays relatively low, confirming the workload is GPU-bound during
  synthesis.
- The audio output is **the same**; only the execution time drops.

> **Takeaway.** Batching improves throughput and GPU utilization, which makes it the
> preferred mode for longer inputs.
"""
)

# ---- 12. Step 6: Visualize the improvement ---------------------------------
md(
"""---

## Step 6 &middot; Visualize the improvement

A picture makes the whole journey obvious. Tool execution time matters across **all
three approaches**, not just sequential vs batched. Moving from cloud **Edge TTS** to
the **local Kokoro** model, and then to **batched** Kokoro, overcomes Edge's
drawbacks and delivers strong execution time entirely on local hardware.
"""
)

img("04_journey.png",
    "Three approaches compared as cards. Edge TTS is the cloud baseline with zero "
    "setup, a 5,000-character cap and text leaving the machine. Kokoro sequential "
    "is the local baseline, one sentence per GPU pass, correct but under-using the "
    "GPU. Kokoro batched is the local optimized approach with many sentences per "
    "GPU pass, length bucketing and masks, same audio at far higher throughput.",
    "The three approaches, from cloud convenience to a fully local, GPU-optimized "
    "tool.",
    width="94%")

md(
"""The cell below uses **Matplotlib** to plot the **tool execution time** of the
three approaches side by side, so the cloud-to-local move and the
sequential-to-batched optimization show up in a single view.

> **Use your own numbers.** `edge_time`, `seq_time` and `batched_time` are
> pre-filled with example values so the chart renders meaningfully before you run
> anything. Replace them with the **execution seconds** from your own runs (each
> tool's output line and the profiling dashboard). Note that for long text Edge
> **truncates** its output, so its time is shown to give context for the cloud
> baseline rather than as a like-for-like comparison.
"""
)

# AMD-branded matplotlib chart (pure presentation, rewritten from orig cell 26).
code(r'''%matplotlib inline

import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- Tool execution time (seconds) for each approach ---
# Default values; replace them with the execution seconds from your own runs,
# taken from each tool's output line and the profiling dashboard.
edge_time = 12.32      # Edge TTS (cloud) - note: truncates long text (~5 min cap)
seq_time = 120.6      # Kokoro, sequential mode (local, unoptimized)
batched_time = 9.62    # Kokoro, batched mode (local, optimized)

for _f in ("Arial", "Liberation Sans", "DejaVu Sans"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break

AMD_RED = "#ED1C24"
INK     = "#1A1A1A"
SUBINK  = "#5B6270"
EDGE_C  = "#F08418"   # cloud Edge (dashboard orange)
SEQ_C   = "#2E6DB4"   # local sequential (dashboard blue)
BATCH_C = "#1EAAB4"   # local batched (teal)

labels = ["Edge\n(cloud)", "Sequential\n(local)", "Batched\n(local, optimized)"]
times  = [edge_time, seq_time, batched_time]
colors = [EDGE_C, SEQ_C, BATCH_C]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.bar(labels, times, color=colors, width=0.62, zorder=3,
              edgecolor="white", linewidth=1.2)
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax.set_ylabel("Tool execution time (seconds)", fontsize=12, color=INK)
ax.set_title("TTS tool execution time: cloud Edge vs local Kokoro",
             fontsize=14, fontweight="bold", color=INK, pad=30)
ax.text(0.5, 1.045, "lower is better", transform=ax.transAxes, ha="center",
        va="bottom", fontsize=10.5, color=SUBINK, style="italic")
ax.set_ylim(0, max(times) * 1.32)
ax.bar_label(bars, fmt="%.1f s", padding=4, fontweight="bold",
             fontsize=12, color=INK)
ax.tick_params(colors=INK, labelsize=11)

notes = []
if seq_time and batched_time:
    notes.append(f"batched is {seq_time / batched_time:.1f}x faster than sequential")
if edge_time and batched_time:
    notes.append(f"batched is {edge_time / batched_time:.1f}x faster than Edge")
if notes:
    ax.text(0.98, 0.96, "\n".join(notes), transform=ax.transAxes,
            ha="right", va="top", fontsize=11.5, fontweight="bold",
            color=AMD_RED,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FDECEC",
                      edgecolor=AMD_RED, linewidth=1.0))

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#C9CDD6")

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/tts_execution_comparison.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()

print(f"Edge: {edge_time:.1f}s | Sequential: {seq_time:.1f}s | "
      f"Batched: {batched_time:.1f}s")''')

# ---- 13. Try it on your own -------------------------------------------------
md(
"""---

## The loop, applied to anything

You have now run the full loop on one example. The same **observability-driven**
workflow applies far beyond TTS.
"""
)

img("03_loop.png",
    "The observability-driven loop as five numbered steps: 1 Run the agent, "
    "2 Fetch in the dashboard, 3 Inspect spans and GPU, 4 Optimize the slow tool, "
    "5 Measure again, with a dashed arrow looping back to step 1 to repeat until "
    "the bottleneck is gone.",
    "The same five moves work for any agent task, not just text-to-speech.",
    width="92%")

md(
"""**Try different inputs for this TTS example**

1. Regenerate `input_text.txt` by rerunning the *Prepare your input text* cell with
   a new topic, or edit the file directly. Longer, multi-sentence text makes the
   batched speed-up more obvious.
2. Re-run the two synthesis cells: `kokoro_tts` in the default (sequential) mode,
   then with `mode='batched'`.
3. In the dashboard, click **Fetch**, select the latest run for each and click
   **Load / Reload**, then update the Matplotlib cell with your own execution
   times to compare.

> **Re-running the Edge baseline.** Once `kokoro_tts` is installed, the agent will
> normally prefer it, so the Edge cell no longer measures Edge. To get a genuine
> Edge run again, remove the custom tool first, then re-run the Step 1 cell:
>
> ```bash
> rm -f ~/.hermes/hermes-agent/tools/kokoro_tts_tool.py       # per-user install
> rm -f /usr/local/lib/hermes-agent/tools/kokoro_tts_tool.py  # container image
> ```
>
> Re-run the *custom tool* install cell above to put it back.

**Try it on your own use cases**

Text-to-speech was only the example. Point the same workflow at *any* Hermes task, a
research query, a coding task, a multi-tool workflow, and:

- **Profile the run** in the dashboard (click **Fetch**, select the run, then
  **Load / Reload**) to see the CPU/GPU timeline and which tool dominated.
- **Inspect the MLflow traces** (the **Traces** view) to drill into each session,
  LLM call, and tool call and see exactly where the time went.
- **Find the bottleneck, optimize it, and measure again**, the same loop you just
  followed here.

> **Tip.** The bigger the workload, the clearer the wins. Let the dashboard and
> traces point you to the slow step instead of guessing.
"""
)

# ---- 14. Conclusion ---------------------------------------------------------
md(
"""---

## Conclusion and key takeaways

This exercise walked through an observability-driven optimization workflow: we
compared cloud-based Edge TTS with a local Kokoro model, identified the bottleneck
through telemetry, optimized the local implementation, and measured the improvement.

| Phase | Tool | Execution | Outcome |
| :--- | :--- | :--- | :--- |
| **1 &middot; Edge TTS** | `text_to_speech` | Cloud-based | Convenient for short inputs, but sends text off-machine and leaves room to reduce execution time. |
| **2 &middot; Kokoro, sequential** | `kokoro_tts` | Local MI300X, one sentence/pass | Complete output, but slower with low GPU utilization. |
| **3 &middot; Kokoro, batched** | `kokoro_tts` (`mode='batched'`) | Local MI300X, many sentences/pass | Same output, significantly better execution time and GPU utilization. |

### What we learned

- **Cloud to local.** Moving from Edge TTS to a local Kokoro model removed the
  input-length limit and kept text processing on the machine.
- **Establish a baseline.** The sequential Kokoro implementation gave us a baseline
  for local TTS performance.
- **Measure, do not guess.** The observability dashboard made it clear that the
  local model was under-using the GPU.
- **Optimize with batching.** A batched mode processes multiple sentences together,
  cutting inference and launch overhead and using the GPU better.
- **Validate the improvement.** Re-running confirmed batched Kokoro beat sequential
  by a wide margin and improved on Edge.
- **Iterate.** Switch to a local model, establish a baseline, find the bottleneck,
  optimize, and measure again.

### Handy extras

- **Raw MLflow UI (advanced).** Browse everything directly at
  `http://<server-ip>:5004`, under the **Traces** and **Runs** tabs.
- **Analysis tab.** Click *Analyze with Hermes* in the dashboard for automatic,
  plain-language suggestions based on the run's `tool_breakdown.csv`.

<details>
<summary><b>Additional information: understanding the kernel cache</b></summary>

<br>

For a specific input using Kokoro:

- **First run (cold run).** The first execution compiles the required GPU kernels.
  This compilation overhead increases execution time.
- **Subsequent runs.** From the second run onward, compiled kernels are reused from
  the cache, giving faster execution and lower latency.
- **Clearing the cache.** For Python-based executions, the kernel cache can be
  cleared with `utils/clear_cache.sh`. In this workshop, however, Kokoro runs as a
  persistent server, so clearing the cache means stopping the server, deleting the
  cache, and restarting before benchmarking again.

**How does the cache work?**

MIOpen stores compiled and tuned GPU kernels in its cache. With
`MIOPEN_FIND_MODE=FAST`, MIOpen skips the expensive kernel search on later runs and
reuses the best cached kernel. COMGR caches the LLVM compilation artifacts, avoiding
recompilation of GPU code on future executions.

**Is the cache reused for every input?**

Not always. The cache is built for specific kernel configurations that depend on
factors such as input shape (sequence length, tensor dimensions). If a new input
needs a configuration that has not been compiled before, MIOpen compiles and caches
it during that execution. Once cached, later runs with the same configuration reuse
it.

</details>
"""
)

# =============================================================================
# WRITE
# =============================================================================
for i, c in enumerate(_cells):
    c["id"] = f"cell-{i:02d}"
_cells = [{k: c[k] for k in sorted(c)} for c in _cells]

nb = {
    "cells": _cells,
    "metadata": {
        # Exactly what Jupyter writes back for this kernel, so a rebuild does not
        # show up as a diff after the notebook has merely been opened and saved.
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"wrote {NB_PATH}")
print(f"cells: {len(_cells)}  "
      f"(md={sum(1 for c in _cells if c['cell_type']=='markdown')}, "
      f"code={sum(1 for c in _cells if c['cell_type']=='code')})")
