"""Standalone Hermes Telemetry Dashboard.

A self-contained Streamlit app for exploring a Hermes agent session recorded in
MLflow by the hermes-otel plugin. Point it at an MLflow tracking server (IP +
port), pick a session id, and it will:

  1. connect to the tracking server and list every session id it can find
     (session_id lives in each trace's trace_metadata - the plugin creates no
     MLflow "run" at all under the plain-OTLP architecture),
  2. resolve the chosen session id -> its experiment,
  3. fetch the session's MLflow traces (one per user turn, full span trees),
  4. for each turn, pull CPU/GPU from Prometheus over that turn's own
     [start, end] window and merge into one session-level CSV (see
     fetch_session_cpu_gpu / save_session_cpu_gpu) - there is no MLflow
     artifact for CPU/GPU anymore, hermes-otel exports them as OTel metrics.

The UI is organized into five tabs:

  * Overview          - CPU% + GPU% timeline with tool-execution spans; a toggle
                        swaps to a full-session span waterfall correlated with
                        CPU/GPU on one shared wall-clock axis.
  * CPU / GPU separate - the two utilization signals on their own charts, with an
                        optional side-by-side raw-CSV panel.
  * Context & tools   - how the agent's context grew call by call across the
                        whole session, per-turn step counts and time-to-first-
                        tool, and every tool outcome including the failures.
  * Traces            - one row per turn (timestamp, latency, tokens, status) with
                        a deep link back into the MLflow trace UI.
  * Analysis          - runs the local ``hermes`` CLI to analyze the session's
                        tool usage and suggest improvements.

Run:
    pip install -r requirements.txt
    streamlit run hermes_profiler.py
"""

import os
import json
import socket
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

import mlflow
from mlflow.tracking import MlflowClient


st.set_page_config(
    page_title="AMD Hermes Telemetry",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# AMD design system
# ---------------------------------------------------------------------------
# One palette drives the CSS below AND every Plotly figure, so the chrome and the
# charts cannot drift apart. These are the same values as
# scripts/build_diagrams.py, which keeps the dashboard, the README diagrams and
# the notebook charts reading as a single design language.

AMD_RED = "#ED1C24"      # brand accent
AMD_RED_DK = "#B3141A"   # gradient end / hover
INK = "#1A1A1A"          # primary text
SUBINK = "#5B6270"       # secondary text
BLUE = "#2E6DB4"         # CPU series
ORANGE = "#F08418"       # tool spans
TEAL = "#1EAAB4"         # local / Kokoro
PANEL = "#F4F5F7"        # secondary surface
LINE = "#D9DCE1"         # hairlines
WHITE = "#FFFFFF"
GREEN = "#2E8B57"        # healthy / success

# Shared Plotly styling. Applying one dict to every figure is what makes the
# charts look designed rather than default.
PLOTLY_LAYOUT = dict(
    font=dict(
        family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
               "Helvetica, Arial, sans-serif",
        size=12,
        color=INK,
    ),
    paper_bgcolor=WHITE,
    plot_bgcolor=WHITE,
    margin=dict(l=56, r=28, t=48, b=44),
    hoverlabel=dict(
        bgcolor=WHITE,
        bordercolor=LINE,
        font=dict(size=12, color=INK),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=11, color=SUBINK),
    ),
    xaxis=dict(
        gridcolor=LINE, griddash="dot", zeroline=False,
        linecolor=LINE, ticks="outside", tickcolor=LINE,
        tickfont=dict(size=11, color=SUBINK),
    ),
    yaxis=dict(
        gridcolor=LINE, griddash="dot", zeroline=False,
        linecolor=LINE, ticks="outside", tickcolor=LINE,
        tickfont=dict(size=11, color=SUBINK),
    ),
)


def style_figure(fig, height=None, axes=True):
    """Apply the AMD chart style to a Plotly figure, in place.

    Called on every figure the app builds so no chart escapes the design system.
    Per-figure settings (titles, ranges, secondary axes) survive because
    update_layout merges rather than replaces.

    axes=False for make_subplots figures: a top-level xaxis/yaxis dict would only
    reach row 1, so those get their axis styling through update_xaxes/update_yaxes
    which correctly fans out to every subplot.
    """
    layout = dict(PLOTLY_LAYOUT)
    axis_style = dict(gridcolor=LINE, griddash="dot", zeroline=False,
                      linecolor=LINE, ticks="outside", tickcolor=LINE,
                      tickfont=dict(size=11, color=SUBINK))
    if not axes:
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
    fig.update_layout(**layout)
    if not axes:
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
    if height:
        fig.update_layout(height=height)
    # Titles read as panel headings rather than chart furniture. Style the title
    # only when the figure actually has one: passing a title dict with no `text`
    # makes Plotly render the literal string "undefined" on untitled figures.
    existing = getattr(fig.layout.title, "text", None)
    if existing:
        fig.update_layout(
            title=dict(text=existing, font=dict(size=14, color=INK),
                       x=0.01, xanchor="left", y=0.97),
        )
    return fig


# Single app-wide style block. Streamlit ships no class hooks, so these target
# stable data-testid attributes. Anything cosmetic lives here rather than being
# sprinkled through the UI code.
st.markdown(
    f"""
    <style>
      :root {{
        --amd-red: {AMD_RED};
        --amd-red-dk: {AMD_RED_DK};
        --ink: {INK};
        --subink: {SUBINK};
        --panel: {PANEL};
        --line: {LINE};
      }}

      html, body,
      [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                       Helvetica, Arial, sans-serif;
          color: var(--ink);
      }}

      /* Reclaim the large default top padding: the branded header should be the
         first thing on screen, not whitespace. */
      [data-testid="stAppViewContainer"] > .main .block-container {{
          padding-top: 2.1rem;
          padding-bottom: 3rem;
          max-width: 1500px;
      }}

      h1, h2, h3, h4 {{ font-weight: 600; letter-spacing: -0.015em; }}
      h1 {{ font-size: 1.9rem; }}

      /* ---------- Branded header ---------- */
      .amd-hero {{
          background: linear-gradient(100deg, #14171C 0%, #23272E 58%, #2C1A1C 100%);
          border-radius: 14px;
          padding: 1.15rem 1.5rem 1.2rem 1.5rem;
          margin-bottom: 1.15rem;
          position: relative;
          overflow: hidden;
          box-shadow: 0 6px 22px rgba(11, 16, 32, 0.20);
      }}
      /* Brand-red edge, the single strongest AMD cue on the page. */
      .amd-hero::before {{
          content: "";
          position: absolute; left: 0; top: 0; bottom: 0;
          width: 5px;
          background: linear-gradient(180deg, var(--amd-red) 0%, var(--amd-red-dk) 100%);
      }}
      .amd-hero-title {{
          color: #FFFFFF;
          font-size: 1.62rem;
          font-weight: 650;
          letter-spacing: -0.02em;
          margin: 0 0 0.22rem 0;
          line-height: 1.2;
      }}
      .amd-hero-title .accent {{ color: var(--amd-red); }}
      .amd-hero-sub {{
          color: #AEB6C2;
          font-size: 0.9rem;
          margin: 0;
          max-width: 76ch;
          line-height: 1.5;
      }}
      .amd-chip-row {{ margin-top: 0.75rem; }}
      .amd-chip {{
          display: inline-block;
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.14);
          color: #E6E9EE;
          font-size: 0.735rem;
          font-weight: 500;
          letter-spacing: 0.02em;
          padding: 0.2rem 0.62rem;
          border-radius: 999px;
          margin-right: 0.4rem;
      }}
      .amd-chip.live {{
          border-color: rgba(46,139,87,0.55);
          color: #8FE0AE;
      }}
      .amd-chip .dot {{
          display: inline-block; width: 6px; height: 6px;
          border-radius: 50%; background: {GREEN};
          margin-right: 0.38rem; vertical-align: middle;
      }}

      /* ---------- KPI cards ---------- */
      /* Bordered containers double as KPI cards; the red top rule ties them to
         the header and gives the row a deliberate dashboard rhythm. */
      [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMetric"]) {{
          background: #FFFFFF;
          border: 1px solid var(--line);
          border-radius: 12px;
          padding: 0.15rem 0.25rem;
          box-shadow: 0 1px 2px rgba(16,24,40,0.04);
          position: relative;
          overflow: hidden;
          transition: box-shadow 140ms ease, transform 140ms ease;
      }}
      [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMetric"])::after {{
          content: "";
          position: absolute; left: 0; right: 0; top: 0; height: 3px;
          background: linear-gradient(90deg, var(--amd-red) 0%, {ORANGE} 100%);
          opacity: 0.9;
      }}
      [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMetric"]):hover {{
          box-shadow: 0 6px 18px rgba(16,24,40,0.09);
          transform: translateY(-1px);
      }}
      [data-testid="stMetricValue"] {{
          font-size: 1.02rem; font-weight: 650; color: var(--ink);
      }}
      [data-testid="stMetricLabel"] {{
          font-weight: 500; color: var(--subink);
          text-transform: uppercase; letter-spacing: 0.045em; font-size: 0.72rem;
      }}

      /* ---------- Tabs ---------- */
      [data-testid="stTabs"] [data-baseweb="tab-list"] {{
          gap: 0.35rem;
          border-bottom: 1px solid var(--line);
      }}
      [data-testid="stTabs"] [data-baseweb="tab"] {{
          height: 42px;
          padding: 0 1.05rem;
          font-weight: 550;
          color: var(--subink);
          border-radius: 8px 8px 0 0;
      }}
      [data-testid="stTabs"] [data-baseweb="tab"]:hover {{
          background: var(--panel); color: var(--ink);
      }}
      [data-testid="stTabs"] [aria-selected="true"] {{ color: var(--amd-red); }}
      [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
          background: var(--amd-red); height: 3px;
      }}

      /* ---------- Sidebar ---------- */
      [data-testid="stSidebar"] {{
          background: linear-gradient(180deg, #FFFFFF 0%, var(--panel) 100%);
          border-right: 1px solid var(--line);
      }}
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
          font-size: 0.83rem;
          text-transform: uppercase;
          letter-spacing: 0.075em;
          color: var(--subink);
          font-weight: 600;
      }}
      /* Inputs: square off the pill shape and light up in brand red on focus. */
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] [data-baseweb="select"] > div {{
          border-radius: 8px !important;
          border-color: var(--line) !important;
      }}
      [data-testid="stSidebar"] input:focus {{
          border-color: var(--amd-red) !important;
          box-shadow: 0 0 0 2px rgba(237,28,36,0.14) !important;
      }}

      /* ---------- Buttons ---------- */
      .stButton > button {{
          border-radius: 8px;
          font-weight: 560;
          border: 1px solid var(--line);
          transition: all 130ms ease;
      }}
      .stButton > button:hover {{
          border-color: var(--amd-red);
          color: var(--amd-red);
      }}
      .stButton > button[kind="primary"] {{
          background: linear-gradient(92deg, var(--amd-red) 0%, var(--amd-red-dk) 100%);
          border: none; color: #FFFFFF;
          box-shadow: 0 2px 8px rgba(237,28,36,0.26);
      }}
      .stButton > button[kind="primary"]:hover {{
          box-shadow: 0 4px 14px rgba(237,28,36,0.36);
          transform: translateY(-1px); color: #FFFFFF;
      }}

      /* ---------- Section rule ---------- */
      /* Small branded heading used above chart blocks. */
      .amd-sec {{
          display: flex; align-items: center; gap: 0.55rem;
          font-size: 0.79rem; font-weight: 650;
          text-transform: uppercase; letter-spacing: 0.075em;
          color: var(--subink);
          margin: 0.35rem 0 0.7rem 0;
      }}
      .amd-sec::before {{
          content: ""; width: 3px; height: 15px; border-radius: 2px;
          background: var(--amd-red);
      }}
      .amd-sec::after {{
          content: ""; flex: 1; height: 1px; background: var(--line);
      }}

      /* Charts sit on a hairline card so they read as panels, not loose ink. */
      [data-testid="stPlotlyChart"] {{
          border: 1px solid var(--line);
          border-radius: 12px;
          padding: 0.45rem 0.3rem 0.2rem 0.3rem;
          background: #FFFFFF;
          box-shadow: 0 1px 2px rgba(16,24,40,0.04);
      }}

      [data-testid="stDataFrame"] {{
          border: 1px solid var(--line); border-radius: 10px;
      }}

      /* Streamlit's default footer/menu add noise to a projected demo. */
      #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def section(label: str):
    """Render a small branded section heading above a chart or table."""
    st.markdown(f'<div class="amd-sec">{label}</div>', unsafe_allow_html=True)

ARTIFACT_DIR = "profiling"

# Cache for a session's downloaded/derived artifacts (CPU/GPU CSVs, traces.json).
# Defaults to profiling_cache/ at the repo root (one level up from this file), so
# it stays inside the project and the dashboard and notebook resolve to the same
# path. Override with the HERMES_PROFILING_CACHE_DIR environment variable.
PROFILING_CACHE_DIR = os.environ.get(
    "HERMES_PROFILING_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "profiling_cache"),
)

# Sidebar logo. The file lives at the REPO ROOT (assets/images/), while this
# module sits in utils/, so a path built only from __file__ + "assets" misses it
# and the logo silently never renders. Check the repo root first, then a
# sibling assets/ dir, so the app works whether it is launched from the repo or
# from a flattened copy (the Docker image puts utils/ and assets/ side by side).
# Override with HERMES_DASHBOARD_LOGO to point at any absolute path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOGO_CANDIDATES = [
    os.environ.get("HERMES_DASHBOARD_LOGO", ""),
    os.path.join(_HERE, os.pardir, "assets", "images", "amd_logo.png"),
    os.path.join(_HERE, "assets", "images", "amd_logo.png"),
]
LOGO_PATH = next(
    (os.path.normpath(p) for p in _LOGO_CANDIDATES if p and os.path.exists(p)),
    os.path.normpath(_LOGO_CANDIDATES[1]),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_tracking_uri(ip: str, port: str) -> str:
    ip = (ip or "").strip()
    port = (port or "").strip()
    if ip.startswith("http://") or ip.startswith("https://"):
        base = ip.rstrip("/")
        return f"{base}:{port}" if port else base
    return f"http://{ip}:{port}"


def _clean_meta(v) -> str:
    """Strip the surrounding JSON quotes MLflow uses for trace_metadata values."""
    s = str(v).strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def _search_traces_all_experiments(client: "MlflowClient", max_results: int = 2000):
    """Yield (experiment, row) for every trace across every experiment.

    mlflow.search_traces() with no location argument only searches the ACTIVE
    experiment (defaults to "Default" and can be empty) - see the same caveat
    in fetch_session_traces() - so this lists every experiment explicitly and
    searches each one.
    """
    experiments = client.search_experiments()
    for exp in experiments:
        try:
            traces = mlflow.search_traces(locations=[exp.experiment_id], max_results=max_results)
        except TypeError:
            traces = mlflow.search_traces(experiment_ids=[exp.experiment_id], max_results=max_results)
        if traces is None or len(traces) == 0:
            continue
        for _, row in traces.iterrows():
            yield exp, row


@st.cache_data(show_spinner=False)
def resolve_run(tracking_uri: str, session_id: str):
    """Find the experiment holding this session's traces, across all experiments.

    hermes-otel's plain OTLP backend does not create an MLflow "run"; a session
    lives purely as trace_metadata["mlflow.trace.session"] on each trace it
    sends, so this resolves a session_id by searching traces, not runs.

    Returns dict(experiment, experiment_id), or None if no trace has that
    session_id. There is no run_id / artifact_uri: nothing downloads MLflow
    artifacts, and CPU/GPU comes from Prometheus (see fetch_session_cpu_gpu).
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    for exp, row in _search_traces_all_experiments(client):
        meta = row.get("trace_metadata")
        if isinstance(meta, dict) and _clean_meta(meta.get("mlflow.trace.session", "")) == session_id:
            return {"experiment": exp.name, "experiment_id": exp.experiment_id}
    return None


@st.cache_data(show_spinner=False)
def fetch_session_ids(tracking_uri: str):
    """Return all distinct session ids across every experiment's traces, newest first.

    session_id lives in trace_metadata["mlflow.trace.session"] (set by the
    plugin on every span it sends), NOT as a run param - there is no run at
    all under the plain-OTLP architecture. Keeps the most recent request_time
    per session id.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    latest = {}  # session_id -> newest request_time seen (epoch ms)
    for _exp, row in _search_traces_all_experiments(client):
        meta = row.get("trace_metadata")
        if not isinstance(meta, dict) or "mlflow.trace.session" not in meta:
            continue
        sid = _clean_meta(meta["mlflow.trace.session"])
        if not sid:
            continue
        try:
            ts = int(float(row.get("request_time")))
        except (TypeError, ValueError):
            ts = 0
        if sid not in latest or ts > latest[sid]:
            latest[sid] = ts
    return [sid for sid, _ in sorted(latest.items(), key=lambda kv: kv[1], reverse=True)]


@st.cache_data(show_spinner=False)
def download_profiling(tracking_uri: str, run_id: str, session_id: str):
    """Download the run's profiling/ artifacts into this session's own folder,
    PROFILING_CACHE_DIR/<session_id>, and return the path to the CSVs. Keeping a
    stable per-session folder (instead of a random temp dir) means the CSVs sit in
    the SAME place as traces.json, so all of a session's telemetry is in one spot.

    Downloading here overwrites the CSVs with the latest, so a resumed session's
    Load reflects the grown timelines; traces.json is untouched (it is written
    separately by the Load handler and is not an MLflow artifact).

    Raises whatever mlflow.artifacts.download_artifacts raises (e.g. on a missing
    artifact or an unreachable server); the caller catches it and shows st.error.
    """
    mlflow.set_tracking_uri(tracking_uri)
    session_dir = os.path.join(PROFILING_CACHE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=ARTIFACT_DIR, dst_path=session_dir
    )


# ---------------------------------------------------------------------------
# CPU / GPU from Prometheus (per-turn, merged into session-level CSVs)
# ---------------------------------------------------------------------------
# hermes-otel exports CPU/GPU as OTel *metrics*, not files - there is no
# profiling/ artifact for them anymore (see host_metrics.py / tracer.py's
# observable gauges). For each turn (trace) in the session, we query Prometheus
# over that turn's own [start, end] window - derived from the trace's own
# spans, the same wall-clock reference the old per-tool CSV used - and merge
# every turn's samples into one session-level CSV. This keeps every other
# function below (read_csv, parse_timestamps, build_figure, start_hermes_analysis)
# working unchanged, since they only care about the CSV files existing in
# local_dir with the legacy column names.

_PROM_MAX_POINTS = 10_000
_PROM_MIN_STEP = 0.1

_TOOL_CSV_HEADER = [
    "turn", "tool_name", "input", "output", "timestamp", "start_time_unix_nano",
    "elapsed_s", "duration_s", "cpu_avg_pct", "cpu_peak_pct", "gpu_avg_pct", "gpu_peak_pct",
]


def _prom_safe_step(duration_s: float, requested_step: float = _PROM_MIN_STEP) -> float:
    """Requested step wins unless the window is long enough to blow past
    Prometheus's per-series point limit, in which case the step scales up."""
    return round(max(requested_step, duration_s / _PROM_MAX_POINTS), 3)


def _prom_query_range(
    prom_url: str, metric: str, start: float, end: float, step: float, instance: str = None
):
    """Raw Prometheus HTTP API call. Returns the list of series, or [] on any
    error, so one turn's metrics being unavailable (e.g. it predates metrics
    being enabled) does not break the whole Load.

    ``instance`` scopes the query to one Hermes process. Without it, any other
    Hermes process reporting the same metric in this time window (for example a
    stale, idle process) is averaged in alongside the real one by
    _merge_gauge_rows, diluting the numbers, because Prometheus has no session
    dimension to filter on otherwise."""
    query = f'{metric}{{instance="{instance}"}}' if instance else metric
    try:
        resp = requests.get(
            f"{prom_url.rstrip('/')}/api/v1/query_range",
            params={"query": query, "start": start, "end": end, "step": step},
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            return []
        return body["data"]["result"]
    except Exception:
        return []


def _trace_instance(trace: dict):
    """The Prometheus `instance` label for the Hermes process that produced this
    trace, read directly off the trace itself.

    hermes-otel does not set service.instance.id explicitly; it inherits the
    OpenTelemetry Python SDK's default Resource behavior, which assigns a random
    UUID once per process (Resource.create() always includes a fresh
    "service.instance.id"). That resource attribute is exported as a tag on every
    trace MLflow stores (info.tags["service.instance.id"]) - the same id that
    becomes Prometheus's `instance` label for every metric that process exports.
    So a session's own trace already identifies its Prometheus instance, and
    filtering on it removes the cross-instance dilution described in
    _prom_query_range. Returns None if the tag is absent, in which case the
    caller falls back to an unfiltered query."""
    return (trace.get("info", {}) or {}).get("tags", {}).get("service.instance.id")


def _turn_window(trace: dict, pad_s: float = 1.0):
    """(turn_number, start_epoch, end_epoch) for one trace, padded by pad_s on
    each side, derived from the trace's own spans (not MLflow's
    request_time/execution_duration) so it lines up with what the plugin
    actually sampled. None if the trace has no usable span timestamps."""
    spans = _norm_spans(trace)
    starts = [s["start"] for s in spans if s["start"] is not None]
    ends = [s["end"] for s in spans if s["end"] is not None]
    if not starts or not ends:
        return None
    return _turn_of_trace(trace), min(starts) / 1e9 - pad_s, max(ends) / 1e9 + pad_s


def _merge_gauge_rows(series, scale: float = 100.0, combine: str = "sum"):
    """Merge a metric's series (one per label combo - e.g. cpu_mode=user/system,
    or one per GPU device) into a single [(ts, value), ...] list, matching how
    the old in-process sampler already combined multi-mode/multi-device
    readings before writing one CSV row: CPU modes summed into a total, GPU
    utilization averaged across devices, GPU power/memory summed across
    devices (see host_metrics.py's Sample.process_cpu_total / gpu_utilization
    and the old csv_dump.py's _on_sample)."""
    if not series:
        return []
    buckets = {}
    for s in series:
        for ts, v in s["values"]:
            buckets.setdefault(ts, []).append(float(v))
    out = []
    for ts, vals in sorted(buckets.items()):
        val = sum(vals) if combine == "sum" else sum(vals) / len(vals)
        out.append((ts, val * scale))
    return out


def _merge_cpu_by_mode(series, scale: float = 100.0):
    out = {}
    for s in series:
        mode = s["metric"].get("cpu_mode", "")
        for ts, v in s["values"]:
            entry = out.setdefault(ts, {"user": 0.0, "system": 0.0, "total": 0.0})
            fv = float(v) * scale
            entry["total"] += fv
            if mode in ("user", "system"):
                entry[mode] += fv
    return out


def _ts_to_str(ts) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def fetch_turn_cpu_gpu(
    prom_url: str, start: float, end: float, step: float = _PROM_MIN_STEP, instance: str = None
):
    """Query all host metrics for one turn's [start, end] window and return
    three DataFrames matching the legacy CSV schemas exactly. ``instance``
    scopes every query to the one real Hermes process for this session (see
    _resolve_session_instance) - without it, any other process reporting in
    this window gets silently averaged in alongside it."""
    step = _prom_safe_step(end - start, step)

    proc_cpu = _merge_cpu_by_mode(
        _prom_query_range(prom_url, "process_cpu_utilization_ratio", start, end, step, instance)
    )
    sys_cpu = _merge_cpu_by_mode(
        _prom_query_range(prom_url, "system_cpu_utilization_ratio", start, end, step, instance)
    )
    gpu_util = _merge_gauge_rows(
        _prom_query_range(prom_url, "hw_gpu_utilization_ratio", start, end, step, instance),
        combine="avg",
    )
    gpu_power = dict(_merge_gauge_rows(
        _prom_query_range(prom_url, "hw_power_watts", start, end, step, instance),
        scale=1.0, combine="sum",
    ))
    gpu_mem = dict(_merge_gauge_rows(
        _prom_query_range(prom_url, "hw_gpu_memory_usage_bytes", start, end, step, instance),
        scale=1.0 / (1024 * 1024), combine="sum",
    ))

    cpu_hermes_df = pd.DataFrame([
        {"timestamp": _ts_to_str(ts), "start_time_unix_nano": int(float(ts) * 1e9),
         "cpu_pct": round(m["total"], 2), "cpu_user_pct": round(m["user"], 2),
         "cpu_system_pct": round(m["system"], 2)}
        for ts, m in sorted(proc_cpu.items())
    ])
    cpu_system_df = pd.DataFrame([
        {"timestamp": _ts_to_str(ts), "start_time_unix_nano": int(float(ts) * 1e9),
         "cpu_pct": round(m["total"], 2), "cpu_user_pct": round(m["user"], 2),
         "cpu_system_pct": round(m["system"], 2)}
        for ts, m in sorted(sys_cpu.items())
    ])
    gpu_df = pd.DataFrame([
        {
            "timestamp": _ts_to_str(ts), "start_time_unix_nano": int(float(ts) * 1e9),
            "gfx_busy_pct": round(v, 2),
            "power_w": round(gpu_power.get(ts, 0.0), 2),
            "vram_mb": round(gpu_mem.get(ts, 0.0), 2),
        }
        for ts, v in gpu_util
    ])
    return cpu_hermes_df, cpu_system_df, gpu_df


def fetch_session_cpu_gpu(prom_url: str, full_traces, step: float = _PROM_MIN_STEP):
    """Per-turn fetch + merge across a whole session. One Prometheus query per
    turn per metric (not one big session-wide query), since a session can span
    long idle gaps between turns that would otherwise blow past Prometheus's
    per-series point limit at fine resolution. Turns whose window returns
    nothing (e.g. they predate metrics being enabled) are skipped, not fatal.

    Every query is scoped to the Prometheus instance each trace's own
    service.instance.id tag names (see _trace_instance); without this, any other
    Hermes process reporting in the same time window is averaged in alongside the
    real one. The instance is resolved per turn, not once for the whole session,
    so it stays correct even when a session's turns ran under more than one
    process."""
    cpu_hermes_parts, cpu_system_parts, gpu_parts = [], [], []
    for trace in (full_traces or []):
        win = _turn_window(trace)
        if win is None:
            continue
        _turn, start, end = win
        instance = _trace_instance(trace)
        ch, cs, gp = fetch_turn_cpu_gpu(prom_url, start, end, step, instance)
        if not ch.empty:
            cpu_hermes_parts.append(ch)
        if not cs.empty:
            cpu_system_parts.append(cs)
        if not gp.empty:
            gpu_parts.append(gp)

    def _combine(parts):
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        # Adjacent turns' padded windows can overlap by ~1s; de-dupe on the
        # exact sample timestamp so a merged session never double-counts a point.
        return (
            df.drop_duplicates(subset=["start_time_unix_nano"])
            .sort_values("start_time_unix_nano")
            .reset_index(drop=True)
        )

    return _combine(cpu_hermes_parts), _combine(cpu_system_parts), _combine(gpu_parts)


def build_tool_execution_df(full_traces):
    """Rebuild tool_execution.csv's rows directly from each tool span's own
    hermes.tool.* attributes - already computed by the plugin from the same
    sampler Prometheus's metrics come from, so no Prometheus query is needed
    for this file; the numbers are embedded in the trace."""
    if not full_traces:
        return pd.DataFrame(columns=_TOOL_CSV_HEADER)

    session_starts = [
        sp["start"] for trace in full_traces for sp in _norm_spans(trace)
        if sp["start"] is not None
    ]
    session_start_ns = min(session_starts) if session_starts else 0

    def _f(v, default=0.0):
        try:
            return float(_clean_attr(v))
        except (TypeError, ValueError):
            return default

    rows = []
    for trace in full_traces:
        turn = _turn_of_trace(trace)
        for sp in _trace_spans(trace):
            name = str(sp.get("name", ""))
            # Identify tool spans structurally (name prefix), NOT by whether
            # hermes.tool.cpu.utilization.avg happened to attach - right after
            # `hermes --resume`, the host-metrics sampler is a brand new
            # thread with no reading yet. A tool call in the resumed
            # process's first ~100ms (host_metrics_interval_ms) can complete
            # before the sampler's first tick, leaving sampler.window() with
            # nothing and the attribute never stamped - even though it's a
            # completely real tool call. Missing avg/peak still default to
            # 0.0 below via _f(), so the row is no longer silently dropped.
            if not name.startswith("tool."):
                continue
            attrs = _span_attrs(sp)
            start = sp.get("start_time_unix_nano") or sp.get("start_time_ns") or sp.get("start_time")
            end = sp.get("end_time_unix_nano") or sp.get("end_time_ns") or sp.get("end_time")
            try:
                start, end = int(start), int(end)
            except (TypeError, ValueError):
                continue
            rows.append({
                "turn": turn if turn is not None else 0,
                "tool_name": name[len("tool."):],
                "input": _clean_attr(attrs.get("input.value", "")),
                "output": _clean_attr(attrs.get("output.value", "")),
                "timestamp": _ts_to_str(start / 1e9),
                "start_time_unix_nano": start,
                "elapsed_s": round((start - session_start_ns) / 1e9, 3),
                "duration_s": round((end - start) / 1e9, 3),
                "cpu_avg_pct": round(_f(attrs.get("hermes.tool.cpu.utilization.avg")) * 100, 2),
                "cpu_peak_pct": round(_f(attrs.get("hermes.tool.cpu.utilization.peak")) * 100, 2),
                "gpu_avg_pct": round(_f(attrs.get("hermes.tool.gpu.utilization.avg")) * 100, 2),
                "gpu_peak_pct": round(_f(attrs.get("hermes.tool.gpu.utilization.peak")) * 100, 2),
            })

    if not rows:
        return pd.DataFrame(columns=_TOOL_CSV_HEADER)
    return (
        pd.DataFrame(rows, columns=_TOOL_CSV_HEADER)
        .sort_values("start_time_unix_nano")
        .reset_index(drop=True)
    )


def save_session_cpu_gpu(local_dir: str, prom_url: str, full_traces):
    """Fetch + merge this session's CPU/GPU from Prometheus (per turn) and its
    tool breakdown from the traces themselves, and write all three CSVs into
    local_dir using the exact legacy filenames/schemas."""
    os.makedirs(local_dir, exist_ok=True)
    cpu_hermes_df, cpu_system_df, gpu_df = fetch_session_cpu_gpu(prom_url, full_traces)
    tool_df = build_tool_execution_df(full_traces)

    cpu_hermes_df.to_csv(os.path.join(local_dir, "cpu_hermes_trace.csv"), index=False)
    cpu_system_df.to_csv(os.path.join(local_dir, "cpu_system_wide.csv"), index=False)
    gpu_df.to_csv(os.path.join(local_dir, "gpu_system_wide.csv"), index=False)
    tool_df.to_csv(os.path.join(local_dir, "tool_execution.csv"), index=False)

    return {
        "cpu_hermes_points": len(cpu_hermes_df),
        "gpu_points": len(gpu_df),
        "tool_rows": len(tool_df),
    }


def traces_cache_path(session_id: str) -> str:
    """On-disk location for a session's full-trace JSON, kept in the SAME folder
    as the downloaded CSVs (PROFILING_CACHE_DIR/<session_id>/profiling) so all of
    a session's telemetry lives in one place."""
    return os.path.join(PROFILING_CACHE_DIR, session_id, ARTIFACT_DIR, "traces.json")


def save_traces_json(session_id: str, full_traces) -> str:
    """Write the fetched full traces to the profiling cache; return the path (or
    "" on failure). Overwrites, so each Load refreshes the on-disk copy."""
    if not session_id or not full_traces:
        return ""
    path = traces_cache_path(session_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(full_traces, f, indent=2, default=str)
        return path
    except Exception:
        return ""


def load_traces_json(session_id: str):
    """Read a session's full traces back from its on-disk file - the single source
    the dashboard renders from - or None if the file does not exist."""
    try:
        with open(traces_cache_path(session_id)) as f:
            return json.load(f)
    except Exception:
        return None


def format_latency_ms(ms) -> str:
    """Format a millisecond duration as ms / s / m, matching MLflow's own style.

    Module-level (not nested in fetch_session_traces) so it can also format the
    session-wide total latency shown at the top of the page. Deliberately
    separate from the waterfall's `_fmt_dur` below: that one takes seconds and
    keeps 2-decimal ms precision (useful for sub-second tool-call spans), while
    this one takes milliseconds and rounds to whole ms (trace-level latency
    doesn't need finer precision).
    """
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return ""
    s = ms / 1000.0
    if s < 1:
        return f"{int(ms)}ms"
    if s < 60:
        return f"{s:.2f}s"
    return f"{s / 60:.2f}m"


@st.cache_data(show_spinner=False)
def fetch_session_traces(tracking_uri: str, session_id: str, experiment_id: str = None):
    """Return a summary DataFrame of MLflow traces for this session.

    Uses mlflow.search_traces (tracing GA in MLflow 2.14+/3.x). Traces are
    matched by the OTel conversation/session id our plugin sets. Returns
    (df, error_message): df is one row per trace (prompt/turn); error_message is
    a string if tracing is unavailable or nothing matched.
    """
    mlflow.set_tracking_uri(tracking_uri)

    def _clean(v):
        # trace_metadata values are JSON-quoted, e.g. '"20260701_..."'. Strip the
        # surrounding quotes for comparison/display.
        s = str(v).strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            s = s[1:-1]
        return s

    try:
        # This MLflow build stores the session in trace_metadata under the key
        # `mlflow.trace.session` (JSON-quoted). It is NOT a server-side filterable
        # attribute here (only request_id/status/timestamp/etc are), so we fetch
        # the experiment's traces and filter client-side on that exact metadata
        # field - never on prompt text, so analysis runs (different session in
        # their own metadata) are correctly excluded.
        # mlflow.search_traces() with no locations searches only the ACTIVE
        # experiment, which defaults to "Default" and is empty here, so the call
        # silently returned nothing and the Turns / Total Latency KPIs sat at
        # n/a even though the run and its traces existed. experiment_id is
        # already resolved by the caller, so scope the search to it explicitly.
        if experiment_id:
            try:
                traces = mlflow.search_traces(
                    experiment_ids=[str(experiment_id)], max_results=2000)
            except TypeError:
                # Newer MLflow renamed experiment_ids to locations.
                traces = mlflow.search_traces(
                    locations=[str(experiment_id)], max_results=2000)
        else:
            traces = mlflow.search_traces(max_results=2000)
    except Exception as e:
        return pd.DataFrame(), f"MLflow tracing not available: {e}"

    if traces is None or len(traces) == 0:
        return pd.DataFrame(), "No traces found."

    df = traces if isinstance(traces, pd.DataFrame) else pd.DataFrame(traces)

    def _session_of(row):
        meta = row.get("trace_metadata", None)
        if isinstance(meta, dict) and "mlflow.trace.session" in meta:
            return _clean(meta["mlflow.trace.session"])
        return None

    if session_id:
        try:
            df = df[df.apply(lambda r: _session_of(r) == session_id, axis=1)]
        except Exception:
            pass
    if df.empty:
        return pd.DataFrame(), "No traces matched this session id."

    base = tracking_uri.rstrip("/")

    def _fmt_ts(v):
        try:
            return pd.to_datetime(int(float(v)), unit="ms").strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return str(v) if v is not None else ""

    def _token_count(row):
        meta = row.get("trace_metadata", None)
        if isinstance(meta, dict) and "mlflow.trace.tokenUsage" in meta:
            try:
                d = json.loads(meta["mlflow.trace.tokenUsage"])
                return d.get("total_tokens")
            except Exception:
                return None
        return None

    rows = []
    total_latency_ms = 0.0
    for _, r in df.iterrows():
        tid = r.get("trace_id")
        # This MLflow build uses `selectedEvaluationId` in the trace-UI URL.
        exp_id = experiment_id
        if exp_id and tid:
            link = f"{base}/#/experiments/{exp_id}/traces?selectedEvaluationId={tid}"
        elif exp_id:
            link = f"{base}/#/experiments/{exp_id}/traces"
        else:
            link = f"{base}/#/traces"
        req = r.get("request")
        raw_latency_ms = r.get("execution_duration")
        try:
            total_latency_ms += float(raw_latency_ms)
        except (TypeError, ValueError):
            pass
        rows.append({
            "trace_id": tid,
            "timestamp": _fmt_ts(r.get("request_time")),
            "latency": format_latency_ms(raw_latency_ms),
            "token_count": _token_count(r),
            "status": str(r.get("state", "")),
            "prompt": (str(req) if req is not None else ""),
            "open_in_mlflow": link,
        })
    summary = pd.DataFrame(rows)
    summary.attrs["raw_columns"] = list(df.columns)
    # Stashed so the caller can show a session-wide total without re-summing the
    # already-formatted per-row "latency" strings.
    summary.attrs["total_latency_ms"] = total_latency_ms
    return summary, ""


@st.cache_data(show_spinner=False)
def fetch_full_traces(tracking_uri: str, trace_ids: tuple):
    """Download the FULL trace JSON (with all spans) for each trace id.

    This mirrors `download_session_traces.py` (`mlflow traces get`): unlike
    fetch_session_traces (which returns a one-row-per-trace summary), this
    returns the complete trace object including every span's inputs/outputs, so
    hermes can see exactly what each tool did. Returns (list_of_trace_dicts,
    error_message).
    """
    mlflow.set_tracking_uri(tracking_uri)
    full = []
    errors = []
    for tid in trace_ids:
        try:
            tr = mlflow.get_trace(tid)
        except Exception as e:
            errors.append(f"{tid}: {e}")
            continue
        # Trace objects expose to_json(); fall back to to_dict() on older builds.
        # Prepend (insert at top) instead of append: trace_ids arrive newest-first
        # (n..1), so inserting each at index 0 yields turn order 1..n in the JSON.
        try:
            full.insert(0, json.loads(tr.to_json()))
        except Exception:
            try:
                full.insert(0, tr.to_dict())
            except Exception as e:
                errors.append(f"{tid}: could not serialize ({e})")
    err = "" if full else ("Could not download full traces: " + "; ".join(errors[:3]))
    return full, err


def read_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Failed to read {os.path.basename(path)}: {e}")
    return pd.DataFrame()


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    # `ts_abs` is the absolute wall-clock axis used to correlate the CSV with the
    # MLflow trace waterfall. When the poller wrote start_time_unix_nano (epoch
    # ns, same reference as MLflow span start_time_unix_nano) we derive it from
    # that, so both signals share one offset-free axis. Older CSVs without that
    # column fall back to the local-time `timestamp` (may be offset from spans).
    if "start_time_unix_nano" in df.columns:
        df["ts_abs"] = pd.to_datetime(
            pd.to_numeric(df["start_time_unix_nano"], errors="coerce"), unit="ns"
        )
    else:
        df["ts_abs"] = df["timestamp"]
    return df.dropna(subset=["timestamp"])


# Alternating tool-span fills, AMD blue / orange at low alpha so the CPU and
# GPU traces stay readable through them.
_SPAN_COLORS = ("rgba(46,109,180,0.13)", "rgba(240,132,24,0.15)")

def _add_tool_spans(fig, tool_df, label_tools=True):
    """Fill each tool's [start, start+duration_s] window as a solid box.

    Consecutive tools alternate between two fill colors so adjacent windows are
    easy to tell apart; each box spans exactly the tool's duration_s window. The
    tool name is labelled at the top of each box.
    """
    if tool_df.empty or "timestamp" not in tool_df.columns:
        return

    idx = 0
    for _, r in tool_df.iterrows():
        start = r["timestamp"]
        if pd.isna(start):
            continue
        try:
            dur = float(r.get("duration_s", 0) or 0)
        except (TypeError, ValueError):
            dur = 0.0
        end = start + timedelta(seconds=max(dur, 0.01))
        name = str(r.get("tool_name", "tool"))

        # Solid alternating fill spanning the whole execution window.
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=_SPAN_COLORS[idx % 2], opacity=1.0, line_width=0,
        )
        if label_tools:
            fig.add_annotation(
                x=start, y=1.0, yref="paper", text=name,
                showarrow=False, textangle=90, xanchor="left", yanchor="top",
                font=dict(size=9, color=SUBINK),
            )
        idx += 1


def build_figure(cpu_df, gpu_df, tool_df) -> go.Figure:
    """CPU% + GPU busy% on a shared 0-100 axis, with tool spans.

    Both signals are normalized to percent of total capacity (CPU is divided by
    the logical core count in the poller), so they share one 0-100 axis and can
    be compared directly.
    """
    fig = go.Figure()

    if not cpu_df.empty and "cpu_pct" in cpu_df.columns:
        fig.add_trace(go.Scatter(
            x=cpu_df["timestamp"], y=cpu_df["cpu_pct"],
            name="CPU % (Hermes + Children)", mode="lines", line=dict(color=BLUE, width=1.8),
            yaxis="y1",
            hovertemplate="CPU %{y:.1f}%<br>%{x|%H:%M:%S.%L}<extra></extra>",
        ))

    if not gpu_df.empty and "gfx_busy_pct" in gpu_df.columns:
        fig.add_trace(go.Scatter(
            x=gpu_df["timestamp"], y=gpu_df["gfx_busy_pct"],
            name="GPU %", mode="lines", line=dict(color=AMD_RED, width=1.8),
            yaxis="y1",
            hovertemplate="GPU %{y:.1f}%<br>%{x|%H:%M:%S.%L}<extra></extra>",
        ))

    _add_tool_spans(fig, tool_df)

    fig.update_layout(
        title="Per-session CPU / GPU utilization with tool spans",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Utilization %", range=[0, 102]),
        hovermode="x unified",
        height=600,
    )
    return style_figure(fig)


def build_single_figure(df, value_col, label, color, y_range=None, tool_df=None,
                        y_title=None):
    """Plot one timeline signal (CPU or GPU) on its own chart with tool spans.

    ``y_title`` overrides the y-axis label; it defaults to ``label`` when unset.
    """
    fig = go.Figure()
    if not df.empty and value_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[value_col],
            name=label, mode="lines", line=dict(color=color, width=1.4),
            hovertemplate=f"{label} %{{y:.1f}}<br>%{{x|%H:%M:%S.%L}}<extra></extra>",
        ))
    if tool_df is not None:
        _add_tool_spans(fig, tool_df)
    yaxis = dict(title=y_title or label, color=color)
    if y_range:
        yaxis["range"] = y_range
    fig.update_layout(
        title=label,
        xaxis=dict(title="Time"),
        yaxis=yaxis,
        hovermode="x unified",
        height=420,
    )
    return style_figure(fig)


def show_left_table(df, height=None):
    """Render df with st.dataframe, left-aligning every column.

    st.dataframe right-aligns numeric columns with no alignment option, so
    numeric columns are formatted as strings (text left-aligns by default).
    The '{:g}' format avoids forced trailing-zero precision.
    """
    disp = df.copy()
    for col in disp.columns:
        if pd.api.types.is_numeric_dtype(disp[col]):
            disp[col] = disp[col].map(lambda v: "" if pd.isna(v) else f"{v:g}")
    kwargs = {"width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(disp, **kwargs)

def _build_fileref_prompt(tool_csv_path: str, traces_path=None) -> str:
    """Prompt that points hermes at the absolute on-disk data files to analyze."""
    abs_tool_csv = os.path.abspath(tool_csv_path)
    abs_traces = os.path.abspath(traces_path) if traces_path else None

    prompt = (
        "You are a senior performance engineer. The telemetry for a Hermes agent session "
        "is on disk. Read these files with your file tools and analyze them "
        "directly. Base your analysis ONLY on their contents.\n\n"
        "**STRICTLY** Do NOT classify tools as CPU-bound or GPU-bound and do NOT "
        "try to attribute the cpu/gpu numbers to a cause - report the wall-clock "
        "timing and the span details, and leave the CPU/GPU interpretation to the "
        "user.\n\n"
        f"1. tool_execution.csv (one row per tool call): {abs_tool_csv}\n"
    )
    if abs_traces:
        prompt += (
            f"2. traces.json (JSON array of full MLflow traces): {abs_traces}\n\n"
            "JOIN KEY: trace span attribute `hermes.turn.number` equals the `turn` column "
            "in tool_execution.csv. Analyze each query using this join key.\n\n"
        )
    prompt += (
        "Please deliver a narrative-driven engineering report using bold markdown headings (e.g., **Heading**) "
        "structured exactly as follows:\n\n"
        "**Session Overview**\n"
        "Provide a summary paragraph followed by a clean Markdown table showing: turn, tool, short key input, "
        "duration (seconds), start offset, and provider details.\n\n"
        "**Executive Summary & Core Metrics**\n\n"
        "For each metric below, write the exact bold metric name followed by the final calculated percentage "
        "on its own line, then insert a BLANK LINE, then write your narrative explanation as a separate "
        "paragraph. The blank line is required so the explanation renders on a new line (a single line break "
        "is not enough in Markdown). Do NOT use a bullet list for these four metrics. Format each exactly like:\n\n"
        "**Success Rate: <pct>%** — computed as `(Successful Root Spans / Total Completed Root Spans) * 100`\n\n"
        "<explanation paragraph on its own line>\n\n"
        "**Tool Selection Accuracy: <pct>%** — computed as `(Valid Schema Calls Without Retries / Total Tool Calls) * 100`\n\n"
        "<explanation paragraph on its own line>\n\n"
        "**Autonomy Score: <pct>%** — computed as `(Autonomous Steps / [Autonomous Steps + Human Interventions]) * 100`\n\n"
        "<explanation paragraph on its own line>\n\n"
        "**Recovery Rate: <pct>%** — computed as `(Errors Followed by Successful Path Correction / Total Errors Encountered) * 100`\n\n"
        "<explanation paragraph on its own line>\n\n"
        "**Per-Query Breakdown**\n"
        "A compact table mapping each query to its total tool time, dominant tool, and outcome, followed by a brief note.\n\n"
        "**Insights**\n"
        "Write your findings as fluid, professional prose. Synthesize the raw numbers into actionable observations. "
        "Drop raw epoch nanoseconds, internal trace IDs, and redundant formula text. Speak directly to execution efficiency, "
        "token scale, and tool behavior."
    )
    return prompt

def start_hermes_analysis(local_dir: str, session_id: str, full_traces=None):
    """Start hermes analysis as a non-blocking subprocess.

    Returns (proc, out_path) or (None, error_message). Output is streamed to a
    temp file so it can be read after the process finishes, and so a blocking
    subprocess.run() never freezes the Streamlit UI and leaves the Stop button
    unclickable. The -z flag gives quiet output.

    The data is always written to disk and referenced by path (never inlined), so
    the prompt stays tiny regardless of how many traces there are and can never
    hit the OS command-line arg limit. tool_execution.csv already sits in
    local_dir (under PROFILING_CACHE_DIR, see download_profiling), so it is
    referenced in place; the traces are reused from the shared per-session cache
    file (traces_cache_path) that Load already wrote, so both the dashboard and
    this analysis read the same traces.json. Everything lives under
    PROFILING_CACHE_DIR so it is purged together on cleanup.
    """
    tool_csv_path = os.path.join(local_dir, "tool_execution.csv")

    traces_path = None
    if full_traces:
        # Reuse the shared per-session cache file (written at Load); only re-save
        # if it's missing, so the analysis and the dashboard use the same file.
        traces_path = traces_cache_path(session_id)
        if not os.path.exists(traces_path):
            traces_path = save_traces_json(session_id, full_traces) or None
        if traces_path is None:
            return None, "Could not stage traces.json for analysis."

    prompt = _build_fileref_prompt(tool_csv_path, traces_path)

    cmd = ["hermes", "--yolo", "-z", prompt, "chat"]
    out_path = os.path.join(tempfile.mkdtemp(prefix="hermes_analysis_"), "out.txt")
    try:
        # Popen duplicates the fd into the child before exec, so the parent's
        # handle can (and should) be closed right after - the child keeps
        # writing to it independently. Using `with` makes that explicit instead
        # of leaving it to be closed whenever the object is garbage collected.
        with open(out_path, "w") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        return proc, out_path
    except FileNotFoundError:
        return None, ("`hermes` CLI not found on this machine. Run the dashboard on "
                      "the same host where hermes is installed.")
    except Exception as e:
        return None, f"Failed to start hermes: {e}"


def read_analysis_output(out_path: str) -> str:
    """Read and clean the hermes analysis output file (may be partial)."""
    try:
        with open(out_path, "r") as f:
            return _strip_hermes_startup(f.read().strip())
    except Exception:
        return ""


def _strip_hermes_startup(raw: str) -> str:
    """Drop the '[hermes-otel] …' plugin startup lines from -z output.

    With -z there is no TUI chrome to parse; only these startup log lines
    precede the answer, so removing them leaves just the response.
    """
    if not raw:
        return ""
    lines = [ln for ln in raw.splitlines() if not ln.lstrip().startswith("[hermes-otel]")]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Trace waterfall (MLflow-style span timeline)
# ---------------------------------------------------------------------------

def _clean_attr(v):
    """Strip the surrounding JSON quotes MLflow uses for attribute values."""
    s = str(v).strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def _span_attrs(span: dict) -> dict:
    """Return a span's attribute dict, tolerating a JSON-string encoding."""
    a = span.get("attributes", {}) if isinstance(span, dict) else {}
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def _trace_spans(trace: dict):
    """Locate the span list inside an MLflow trace dict (schema-tolerant)."""
    if not isinstance(trace, dict):
        return []
    data = trace.get("data")
    if isinstance(data, dict) and isinstance(data.get("spans"), list):
        return data["spans"]
    if isinstance(trace.get("spans"), list):
        return trace["spans"]
    return []


def _turn_of_trace(trace: dict):
    """Best-effort turn number for a trace, read from the hermes.turn.number
    span attribute the plugin sets. Returns int, str, or None."""
    for sp in _trace_spans(trace):
        a = _span_attrs(sp)
        if "hermes.turn.number" in a:
            raw = _clean_attr(a["hermes.turn.number"])
            try:
                return int(raw)
            except (TypeError, ValueError):
                return raw
    return None


def _fmt_dur(sec: float) -> str:
    """Human-readable duration matching MLflow's style (ms / s / m).

    Sibling of `format_latency_ms` above, kept separate because this one takes
    seconds and needs sub-ms precision for short tool-call spans.
    """
    if sec < 1:
        return f"{sec * 1000:.2f}ms"
    if sec < 60:
        return f"{sec:.2f}s"
    return f"{sec / 60:.2f}m"


def _norm_spans(trace: dict):
    """Normalize raw MLflow spans to {span_id, parent_id, name, start, end, type}.

    start/end are integer nanoseconds. Field names vary across MLflow builds, so
    each is looked up under several possible keys.
    """
    out = []
    for sp in _trace_spans(trace):
        if not isinstance(sp, dict):
            continue
        ctx = sp.get("context", {}) if isinstance(sp.get("context"), dict) else {}
        sid = sp.get("span_id") or ctx.get("span_id")
        pid = sp.get("parent_id") or sp.get("parent_span_id") or ctx.get("parent_id")

        def _pick(*keys):
            for k in keys:
                v = sp.get(k)
                if v is not None:
                    return v
            return None

        start = _pick("start_time", "start_time_ns", "start_time_unix_nano")
        end = _pick("end_time", "end_time_ns", "end_time_unix_nano")
        try:
            start = int(start)
        except (TypeError, ValueError):
            start = None
        try:
            end = int(end)
        except (TypeError, ValueError):
            end = None

        attrs = _span_attrs(sp)
        stype = _clean_attr(attrs.get("mlflow.spanType", "")) or ""
        out.append({
            "span_id": sid, "parent_id": pid,
            "name": str(sp.get("name", "span")),
            "start": start, "end": end, "type": stype,
        })
    return out


def _order_spans(spans):
    """Return spans in tree (DFS) order with a `_depth` key on each.

    Roots are spans whose parent id is missing or points outside this trace;
    siblings are ordered by start time so the waterfall reads top-to-bottom in
    execution order.
    """
    ids = {s["span_id"] for s in spans}
    children, roots = {}, []
    for s in spans:
        pid = s["parent_id"]
        if pid and pid in ids:
            children.setdefault(pid, []).append(s)
        else:
            roots.append(s)

    def _k(s):
        return s["start"] if s["start"] is not None else 0

    ordered = []

    def _dfs(s, depth):
        s["_depth"] = depth
        ordered.append(s)
        for c in sorted(children.get(s["span_id"], []), key=_k):
            _dfs(c, depth + 1)

    for r in sorted(roots, key=_k):
        _dfs(r, 0)
    return ordered


# Waterfall bar colors, keyed to the AMD palette: model work in blue, tool work
# in orange, retrieval/chain in teal, parsing in brand red.
_SPAN_TYPE_COLORS = {
    "LLM": BLUE, "CHAT_MODEL": BLUE, "AGENT": BLUE,
    "TOOL": ORANGE,
    "CHAIN": TEAL, "RETRIEVER": TEAL,
    "PARSER": AMD_RED, "RERANKER": AMD_RED,
}
_SPAN_DEFAULT_COLOR = "#8899A6"


def build_session_waterfall_figure(traces, cpu_df, gpu_df, tool_df=None) -> go.Figure:
    """Full-session span waterfall (ALL turns) stacked over the CPU/GPU timeline,
    on one shared absolute wall-clock x-axis.

    Each span is placed at its ABSOLUTE position from start_time_unix_nano (epoch
    ns), the same reference the poller writes into cpu_hermes_trace.csv /
    gpu_system_wide.csv (ts_abs).
    That puts the spans and the utilization lines on one axis, so the idle wait
    between two queries shows up as the same blank gap in both. A per-trace
    waterfall would instead re-base each span to its own trace start, showing one
    turn at a time and hiding that gap.
    """
    def _key(tr):
        t = _turn_of_trace(tr)
        if isinstance(t, int):
            return (0, t)
        ns = [s["start"] for s in _norm_spans(tr) if s["start"] is not None]
        return (1, min(ns) if ns else 0)

    ordered = sorted(traces or [], key=_key)

    ys, bases, widths, texts, ticktext, colors, hovers = [], [], [], [], [], [], []
    turn_marks = []  # (turn_label, first_span_start_dt), one per turn, for dividers
    y = 0
    for tr in ordered:
        turn = _turn_of_trace(tr)
        spans = [s for s in _order_spans(_norm_spans(tr)) if s["start"] is not None]
        if not spans:
            continue
        turn_marks.append((turn, pd.to_datetime(min(s["start"] for s in spans), unit="ns")))
        for s in spans:
            start_dt = pd.to_datetime(s["start"], unit="ns")
            if s["end"] is not None:
                dur = max((s["end"] - s["start"]) / 1e9, 1e-4)
            else:
                dur = 1e-4
            ys.append(y)
            bases.append(start_dt)
            # go.Bar Gantt pattern on a date axis: base=datetime start, width in
            # MILLISECONDS as a plain number. This only honors `base` when the
            # x-axis is explicitly type="date" (set below on both rows); without
            # that, Plotly drops `base` and stacks every bar at the left edge.
            widths.append(dur * 1000.0)
            texts.append(_fmt_dur(dur))
            ticktext.append((" " * 4 * s["_depth"]) + s["name"])
            colors.append(_SPAN_TYPE_COLORS.get((s["type"] or "").upper(), _SPAN_DEFAULT_COLOR))
            hovers.append(
                f"<b>{s['name']}</b><br>turn: {turn}<br>type: {s['type'] or 'n/a'}"
                f"<br>start: {start_dt.strftime('%H:%M:%S.%f')[:-3]}"
                f"<br>duration: {_fmt_dur(dur)}"
            )
            y += 1

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38], vertical_spacing=0.06,
    )

    if not ys:
        fig.update_layout(height=200, title="No spans found across this session's traces.")
        return style_figure(fig, axes=False)

    fig.add_trace(go.Bar(
        x=widths, base=bases, y=ys, orientation="h",
        marker=dict(color=colors), text=texts, textposition="outside",
        hovertext=hovers, hoverinfo="text", cliponaxis=False, showlegend=False,
    ), row=1, col=1)

    if not cpu_df.empty and "cpu_pct" in cpu_df.columns and "ts_abs" in cpu_df.columns:
        fig.add_trace(go.Scatter(
            x=cpu_df["ts_abs"], y=cpu_df["cpu_pct"],
            name="CPU %", mode="lines", line=dict(color=BLUE, width=1.8),
            hovertemplate="CPU %{y:.1f}%<br>%{x|%H:%M:%S.%L}<extra></extra>",
        ), row=2, col=1)
    if not gpu_df.empty and "gfx_busy_pct" in gpu_df.columns and "ts_abs" in gpu_df.columns:
        fig.add_trace(go.Scatter(
            x=gpu_df["ts_abs"], y=gpu_df["gfx_busy_pct"],
            name="GPU %", mode="lines", line=dict(color=AMD_RED, width=1.8),
            hovertemplate="GPU %{y:.1f}%<br>%{x|%H:%M:%S.%L}<extra></extra>",
        ), row=2, col=1)

    # Shade each tool's [start, start+duration_s] window on the utilization row
    # with alternating translucent fills, marking which tool drove the CPU/GPU
    # activity. Uses tool_df["ts_abs"], which parse_timestamps derives from
    # tool_execution.csv's start_time_unix_nano column - the same epoch-ns
    # reference as the CPU/GPU CSVs and the MLflow spans, so the shading lines up
    # regardless of the host's timezone. Older CSVs without that column fall back
    # to the local-time string and may be offset on a non-UTC host.
    if tool_df is not None and not tool_df.empty and "ts_abs" in tool_df.columns:
        _tcolors = _SPAN_COLORS
        j = 0
        for _, r in tool_df.iterrows():
            ts = r.get("ts_abs")
            if pd.isna(ts):
                continue
            try:
                d = float(r.get("duration_s", 0) or 0)
            except (TypeError, ValueError):
                d = 0.0
            fig.add_vrect(
                x0=ts, x1=ts + pd.Timedelta(seconds=max(d, 0.01)),
                fillcolor=_tcolors[j % 2], opacity=1.0, line_width=0,
                layer="below", row=2, col=1,
            )
            # Label the box with the tool name (rotated); add_vrect draws only the
            # color fill, so the name needs its own annotation.
            fig.add_annotation(
                x=ts, y=100, row=2, col=1,
                text=str(r.get("tool_name", "tool")),
                showarrow=False, textangle=90, xanchor="left", yanchor="top",
                font=dict(size=9, color=SUBINK),
            )
            j += 1

    # Dotted divider + label at each turn's first span, spanning both rows so a
    # span in the top chart lines up with its CPU/GPU footprint below.
    for turn, start_dt in turn_marks:
        try:
            fig.add_vline(
                x=start_dt,
                line=dict(color="rgba(120,120,120,0.45)", width=1, dash="dot"),
                annotation_text=(f"Turn {turn}" if turn is not None else "Turn"),
                annotation_position="top",
                annotation_font=dict(size=10, color=SUBINK),
            )
        except Exception:
            pass

    fig.update_layout(
        height=max(520, 20 * len(ys) + 260),
        hovermode="x unified",
        bargap=0.3,
        margin=dict(l=10, r=50, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(
        autorange="reversed", tickmode="array", tickvals=ys, ticktext=ticktext,
        tickfont=dict(family="monospace", size=10), row=1, col=1,
    )
    fig.update_yaxes(title_text="Utilization %", range=[0, 102], row=2, col=1)
    # Both rows must be date axes (and matched) so the bars line up with the
    # timeline. shared_xaxes matches the range; type=date must be set on both.
    fig.update_xaxes(type="date", row=1, col=1)
    fig.update_xaxes(title_text="Wall-clock time", type="date", row=2, col=1)
    return style_figure(fig, axes=False)

# ---------------------------------------------------------------------------
# Context & tool metrics
# ---------------------------------------------------------------------------
# Four metrics derived from the span tree the dashboard already downloads at
# Load: agent steps, time-to-first-tool, context growth per step, and tool
# failure rate. Nothing here needs new instrumentation or a re-run -- it is
# arithmetic over telemetry the patched hermes-otel plugin already emits, so it
# also works on sessions recorded before these metrics existed.
#
# These build on the span helpers above (_clean_attr / _span_attrs /
# _trace_spans) rather than re-implementing them.

# Bumped whenever compute_session_metrics gains or renames a key. The Load
# handler stashes its result in st.session_state, which survives a code reload,
# so a dict built by an older version would otherwise be read by newer render
# code and raise KeyError on a key that did not exist yet.
METRICS_SCHEMA = 4

NS_PER_S = 1_000_000_000

# Mirrors the plugin's own policy in on_post_tool_call: only genuine execution
# failures count against the failure rate. A blocked or denied call is a policy
# or human decision, not a performance defect, so it is tracked separately
# rather than inflating the headline number.
FAILURE_OUTCOMES = frozenset({
    "error", "failed", "failure", "exception", "timeout", "timed_out",
})
NEUTRAL_OUTCOMES = frozenset({
    "blocked", "denied", "rejected", "cancelled", "canceled", "skipped",
    "interrupted",
})


def _decode_str(value):
    """Fully decode a string span attribute, unescaping JSON escapes.

    _clean_attr only strips the surrounding quotes, which is enough for short
    labels but leaves "\\n" as two literal characters -- so a multi-line value
    like a traceback comes back as one long line and splitlines() finds nothing
    to split. Anything inspecting the *content* of a string attribute needs this.
    """
    if not isinstance(value, str):
        return "" if value is None else str(value)
    text = value.strip()
    if text[:1] == '"' and text[-1:] == '"':
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str):
                return decoded
        except (ValueError, TypeError):
            pass
    return _clean_attr(value)


def _metric_int(value):
    """Best-effort int, tolerating JSON-quoted numbers and None."""
    if value is None:
        return None
    try:
        return int(float(_clean_attr(value)))
    except (TypeError, ValueError):
        return None


def _classify_span(name, attrs):
    """Bucket a span into agent / api / llm_turn / tool / other.

    Name prefix is checked before mlflow.spanType deliberately. Hermes emits two
    span kinds that both carry spanType == "LLM": one llm.<model> span wrapping
    the whole turn, and one api.<model> span per HTTP round trip. Only the name
    separates them, and conflating the two would make every task look like a
    single step. The spanType fallback keeps this usable against other
    frameworks, which typically emit one LLM span per call.
    """
    lowered = (name or "").lower()
    if lowered.startswith("api."):
        return "api"
    if lowered.startswith("tool."):
        return "tool"
    if lowered.startswith("llm."):
        return "llm_turn"
    if lowered == "agent" or lowered.startswith("agent."):
        return "agent"
    stype = _clean_attr(attrs.get("mlflow.spanType", "")).upper()
    if stype == "TOOL":
        return "tool"
    if stype in ("LLM", "CHAT_MODEL"):
        return "api"
    if stype == "AGENT":
        return "agent"
    return "other"


def _metric_spans(trace):
    """Flatten a trace into {kind, name, start, end, dur_s, attrs} records.

    Spans without a usable start timestamp are dropped: every metric here is
    positional, so an unplaceable span cannot contribute and keeping it would
    corrupt counts. Returned in start order.
    """
    out = []
    for sp in _trace_spans(trace):
        if not isinstance(sp, dict):
            continue
        start = end = None
        for key in ("start_time", "start_time_ns", "start_time_unix_nano"):
            if sp.get(key) is not None:
                start = _metric_int(sp.get(key))
                if start is not None:
                    break
        if start is None:
            continue
        for key in ("end_time", "end_time_ns", "end_time_unix_nano"):
            if sp.get(key) is not None:
                end = _metric_int(sp.get(key))
                if end is not None:
                    break
        attrs = _span_attrs(sp)
        name = str(sp.get("name", ""))
        out.append({
            "kind": _classify_span(name, attrs),
            "name": name,
            "start": start,
            "end": end,
            "dur_s": ((end - start) / NS_PER_S) if end is not None else 0.0,
            "attrs": attrs,
            "status": sp.get("status", ""),
        })
    out.sort(key=lambda x: x["start"])
    return out


def _tool_outcome(span):
    """Outcome label for one tool span.

    hermes.tool.outcome is set per call by the plugin and is authoritative. When
    absent -- an unpatched plugin, or another framework -- fall back to the same
    rules the plugin's extract_tool_result_status applies to the raw payload, so
    both paths agree on what counts as a failure.
    """
    attrs = span.get("attrs", {})
    outcome = attrs.get("hermes.tool.outcome")
    if outcome:
        return _clean_attr(outcome).lower()

    raw = attrs.get("output.value", attrs.get("mlflow.spanOutputs"))
    if raw is None:
        # An OTel ERROR status is a weaker signal than the payload, but better
        # than declaring success with no evidence either way.
        return "error" if str(span.get("status", "")).upper() == "ERROR" else "unknown"

    payload = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            payload = raw
    if isinstance(payload, dict):
        status = payload.get("status")
        if isinstance(status, str) and status.strip():
            return status.strip().lower()
        if payload.get("error") and str(payload["error"]).strip():
            return "error"
        if payload.get("timeout"):
            return "timeout"
        if payload.get("blocked"):
            return "blocked"
    return "completed"


def _error_summary(span, limit=160):
    """One-line reason a tool call failed, or "" if none can be extracted.

    Prefers the plugin's error.message attribute. Many tools instead report
    failure as {"status": "error", "output": "<traceback>"} with no error key, so
    the fallback takes the LAST non-empty line -- for a Python traceback that is
    the exception line (KeyError: 'turn'), the one line worth showing in a table.
    """
    attrs = span.get("attrs", {})
    msg = attrs.get("error.message")
    if msg:
        lines = [ln.strip() for ln in _decode_str(msg).splitlines() if ln.strip()]
        if lines:
            return lines[-1][:limit]

    raw = attrs.get("output.value", attrs.get("mlflow.spanOutputs"))
    if raw is None:
        return ""
    payload = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            payload = raw
    text = ""
    if isinstance(payload, dict):
        for key in ("error", "output", "stderr", "message"):
            value = payload.get(key)
            if value and str(value).strip():
                text = str(value)
                break
    else:
        text = str(payload)
    lines = [ln.strip() for ln in _decode_str(text).splitlines() if ln.strip()]
    return lines[-1][:limit] if lines else ""


def _slope(values):
    """Least-squares slope of values against their 0-based index.

    Reported as "tokens added per step" so the number stays comparable whether a
    task took 4 steps or 25. Uses every point, so one large jump cannot dominate
    the way it would with (last - first) / (n - 1). Returns 0.0 below 2 points.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return (num / den) if den else 0.0


def _in_tokens(attrs):
    for key in ("gen_ai.usage.input_tokens", "llm.token_count.prompt",
                "llm.request.approx_input_tokens"):
        value = _metric_int(attrs.get(key))
        if value is not None:
            return value
    return None


def _out_tokens(attrs):
    for key in ("gen_ai.usage.output_tokens", "llm.token_count.completion"):
        value = _metric_int(attrs.get(key))
        if value is not None:
            return value
    return None


def compute_turn_metrics(trace):
    """Compute the four metrics for a single turn (one MLflow trace).

    Metrics that cannot be computed are None rather than 0 -- a task that called
    no tools has *no* time-to-first-tool, which differs from "zero seconds", and
    collapsing the two drags any average toward a value never observed.
    """
    spans = _metric_spans(trace)
    result = {
        "turn": None, "wall_s": None, "start_ns": None,
        "time_to_first_tool_s": None, "time_to_first_tool_pct": None,
        "agent_steps": 0, "agent_steps_reported": None,
        "context_series": [], "context_first": None, "context_last": None,
        "context_delta": None, "context_growth_per_step": None,
        "context_steps": [],
        "input_tokens_total": None, "output_tokens_total": None,
        "tool_calls": 0, "tool_failures": 0, "tool_neutral": 0,
        "tool_failure_rate_pct": None, "tool_time_failed_s": 0.0,
        "outcome_counts": {}, "first_tool_name": None,
        "outcomes_reported": [], "hidden_failure": False,
        "failed_tools": [], "failed_calls": [],
    }
    if not spans:
        return result

    api = [x for x in spans if x["kind"] == "api"]
    tools = [x for x in spans if x["kind"] == "tool"]
    llm_turn = [x for x in spans if x["kind"] == "llm_turn"]

    turn_start = min(x["start"] for x in spans)
    turn_end = max((x["end"] for x in spans if x["end"] is not None),
                   default=turn_start)
    wall_s = (turn_end - turn_start) / NS_PER_S
    result["wall_s"] = wall_s
    # Kept so turns can be ordered even when hermes.turn.number is missing --
    # the session-wide context curve is only meaningful in turn order.
    result["start_ns"] = turn_start

    for span in spans:
        if "hermes.turn.number" in span["attrs"]:
            result["turn"] = _metric_int(span["attrs"]["hermes.turn.number"])
            break

    # -- Agent steps --------------------------------------------------------
    # Count api spans; fall back to per-call LLM spans for frameworks emitting
    # only those. The plugin's own counter rides alongside as a cross-check, so
    # an unexported span shows as a disagreement rather than silently lowering
    # the count.
    result["agent_steps"] = len(api) if api else len(llm_turn)
    for span in spans:
        reported = _metric_int(span["attrs"].get("hermes.turn.api_call_count"))
        if reported is not None:
            result["agent_steps_reported"] = reported
            break

    # -- Time to first tool -------------------------------------------------
    if tools:
        first = min(tools, key=lambda x: x["start"])
        ttft = (first["start"] - turn_start) / NS_PER_S
        result["time_to_first_tool_s"] = round(ttft, 3)
        result["first_tool_name"] = first["name"].replace("tool.", "", 1)
        if wall_s > 0:
            result["time_to_first_tool_pct"] = round(ttft / wall_s * 100, 1)

    # -- Context growth per step -------------------------------------------
    series = [t for t in (_in_tokens(x["attrs"]) for x in api) if t is not None]
    if series:
        result["context_series"] = series
        result["context_first"] = series[0]
        result["context_last"] = series[-1]
        result["context_delta"] = series[-1] - series[0]
        result["context_growth_per_step"] = round(_slope([float(v) for v in series]), 1)
        result["input_tokens_total"] = sum(series)
    outs = [t for t in (_out_tokens(x["attrs"]) for x in api) if t is not None]
    if outs:
        result["output_tokens_total"] = sum(outs)

    # Pair each LLM call with the tool it asked for, so a jump in the context
    # curve can be attributed to the tool result that caused it. The tool a call
    # requested is the first tool span starting after that call ends and before
    # the next call begins -- the agent is strictly serial, so that window holds
    # exactly one tool.
    steps = []
    prev_in = None
    for i, span in enumerate(api):
        after = span["end"] if span["end"] is not None else span["start"]
        before = api[i + 1]["start"] if i + 1 < len(api) else None
        tool_name = None
        for tspan in tools:
            if tspan["start"] >= after and (before is None or tspan["start"] < before):
                tool_name = tspan["name"].replace("tool.", "", 1)
                break
        tokens_in = _in_tokens(span["attrs"])
        finish = _clean_attr(span["attrs"].get("llm.response.finish_reason", "")) or None
        steps.append({
            "step": i + 1,
            "input_tokens": tokens_in,
            "output_tokens": _out_tokens(span["attrs"]),
            # None on step 1: there is no previous call to compare against, and
            # 0 would read as "nothing was added".
            "delta": (tokens_in - prev_in
                      if tokens_in is not None and prev_in is not None else None),
            "tool": tool_name,
            "finish_reason": finish,
            "llm_s": round(span["dur_s"], 2),
        })
        if tokens_in is not None:
            prev_in = tokens_in
    result["context_steps"] = steps

    # -- Tool failure rate --------------------------------------------------
    counts = {}
    failures = neutral = 0
    failed_time = 0.0
    failed_tools = []
    failed_calls = []
    for span in tools:
        label = _tool_outcome(span)
        counts[label] = counts.get(label, 0) + 1
        if label in FAILURE_OUTCOMES:
            failures += 1
            failed_time += span["dur_s"]
            name = span["name"].replace("tool.", "", 1)
            # A list, not a set: the same tool failing twice in a row is the
            # signal that the agent is stuck rather than unlucky, and a set
            # would erase exactly that.
            failed_tools.append(name)
            failed_calls.append({
                "tool": name,
                "outcome": label,
                "offset_s": round((span["start"] - turn_start) / NS_PER_S, 2),
                "duration_s": round(span["dur_s"], 3),
                "error": _error_summary(span),
            })
        elif label in NEUTRAL_OUTCOMES:
            neutral += 1
    result["tool_calls"] = len(tools)
    result["tool_failures"] = failures
    result["tool_neutral"] = neutral
    result["tool_time_failed_s"] = round(failed_time, 3)
    result["outcome_counts"] = counts
    result["failed_tools"] = failed_tools
    result["failed_calls"] = failed_calls

    # Cross-check the per-span view against the plugin's per-turn outcome set,
    # and flag a disagreement rather than reporting a rate known to be low.
    #
    # This is not hypothetical. The plugin keys both the tool span and the CSV
    # row on f"{tool_name}:{task_id}", so when a model calls the same tool twice
    # inside one step -- a failed attempt then a retry -- the two collide and
    # only one survives. The set records which outcomes occurred but not how
    # many, so the hidden call can be detected but not counted. Reporting
    # "0% observed, and I know I am blind" beats a bare 0%.
    reported = []
    for span in spans:
        raw = span["attrs"].get("hermes.turn.tool_outcomes")
        if raw:
            reported = [p.strip().lower()
                        for p in _clean_attr(raw).split(",") if p.strip()]
            break
    result["outcomes_reported"] = reported
    if reported:
        hidden = {o for o in reported if o in FAILURE_OUTCOMES}
        observed = {o for o, n in counts.items() if n > 0}
        result["hidden_failure"] = bool(hidden - observed)

    if tools:
        result["tool_failure_rate_pct"] = round(failures / len(tools) * 100, 1)
    return result


def _mean(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None




def compute_session_metrics(traces):
    """Aggregate the four metrics across every turn in a session.

    Rates are pooled over raw counts rather than averaged over per-turn rates: a
    turn with one tool call and a turn with twenty should not carry equal weight.
    """
    turns = [compute_turn_metrics(t) for t in (traces or [])]
    turns = [t for t in turns if t["wall_s"] is not None]

    total_tools = sum(t["tool_calls"] for t in turns)
    total_failures = sum(t["tool_failures"] for t in turns)

    summary = {
        "schema": METRICS_SCHEMA,
        "turns": len(turns),
        "per_turn": turns,

        "agent_steps_total": sum(t["agent_steps"] for t in turns),
        "agent_steps_mean": _mean(t["agent_steps"] for t in turns),
        "agent_steps_max": max((t["agent_steps"] for t in turns), default=None),

        "time_to_first_tool_mean_s": _mean(t["time_to_first_tool_s"] for t in turns),
        "time_to_first_tool_max_s": max(
            (t["time_to_first_tool_s"] for t in turns
             if t["time_to_first_tool_s"] is not None), default=None),
        "time_to_first_tool_pct_mean": _mean(t["time_to_first_tool_pct"] for t in turns),

        "context_growth_per_step_mean": _mean(
            t["context_growth_per_step"] for t in turns),
        "context_peak": max((t["context_last"] for t in turns
                             if t["context_last"] is not None), default=None),
        "input_tokens_total": sum(t["input_tokens_total"] or 0 for t in turns) or None,
        "output_tokens_total": sum(t["output_tokens_total"] or 0 for t in turns) or None,

        "tool_calls_total": total_tools,
        "tool_failures_total": total_failures,
        "tool_failure_rate_pct": (round(total_failures / total_tools * 100, 1)
                                  if total_tools else None),
        "tool_time_failed_s": round(sum(t["tool_time_failed_s"] for t in turns), 3),
        # Turns where the plugin recorded a failure that no span captured. A
        # non-zero count means tool_failure_rate_pct is a lower bound.
        "hidden_failure_turns": sum(1 for t in turns if t["hidden_failure"]),
    }

    outcomes = {}
    failed_counts = {}
    all_failed_calls = []
    for turn in turns:
        for label, n in turn["outcome_counts"].items():
            outcomes[label] = outcomes.get(label, 0) + n
        for name in turn["failed_tools"]:
            failed_counts[name] = failed_counts.get(name, 0) + 1
        for call in turn["failed_calls"]:
            all_failed_calls.append(dict(call, turn=turn["turn"]))
    summary["outcome_counts"] = outcomes
    # Worst offender first -- that is the one worth fixing.
    summary["failed_tool_counts"] = dict(
        sorted(failed_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    summary["failed_calls"] = all_failed_calls
    return summary


# ---------------------------------------------------------------------------
# Continuous context-growth curve
# ---------------------------------------------------------------------------
# The context an agent carries into turn N+1 is the context it left turn N with,
# so this is one curve for the whole session rather than one line per turn.
# Every LLM call in the session gets a step index that keeps counting across turn
# boundaries (turn 1 -> 1..3, turn 2 -> 4..6, turn 3 -> 7..10), consecutive turns
# are bridged, and the turn-local step number moves into the hover box.

# Line color per turn, cycled. The turn label in the hover box (not the color)
# is what identifies a point, so repeating after eight turns costs nothing.
TURN_COLORS = [BLUE, ORANGE, TEAL, AMD_RED, GREEN, "#7A5AF8", AMD_RED_DK, SUBINK]


def _ordered_turn_metrics(per_turn):
    """per_turn in wall-clock order, by turn start.

    Sorting by hermes.turn.number looks right and is wrong: the agent restarts
    its turn counter after a `--resume`, so ONE session can hold turn 1 several
    times (observed: turns 1,1,2,3,4,5,6,1,2 in a single 9-trace session).
    Ordering by that number then drags a late turn-1 trace back next to the
    first one and renumbers the session away from the order it actually ran in,
    which drew a curve that climbed to 29k and dropped back to 19k at the turn-2
    boundary. start_ns is monotonic whatever the agent calls its turns, so the
    curve reads left-to-right in execution order and never steps backwards.
    """
    def _key(item):
        i, turn = item
        start = turn.get("start_ns")
        return (0, start, i) if start is not None else (1, 0, i)

    return [t for _, t in sorted(enumerate(per_turn or []), key=_key)]


def context_growth_points(per_turn):
    """Flatten every turn's LLM calls into one continuously-numbered series.

    Each point carries `x`, a session-wide step index that keeps counting across
    turns, alongside `step`, the step number within its own turn -- so the curve
    is continuous while the hover box can still say "Turn 2 - step 1".

    `delta` is measured against the previous point in the SESSION rather than in
    the turn, so the first step of a turn reports the jump carried over from the
    turn before it instead of a blank. `new_turn` marks exactly those points, so
    the hover can name where the jump came from.
    """
    points = []
    prev = None
    # Turns are numbered by their POSITION in the session, 1..N, not by the
    # agent's own hermes.turn.number: that counter restarts after a `--resume`,
    # so one session can report turn 1 three times. Renumbering keeps the
    # legend and the bands reading 1,2,3,... in the order the turns actually
    # ran, and keeps every turn its own group -- a repeated number would
    # otherwise merge two separate runs of points into a single line and draw
    # the curve stepping backwards.
    for i, turn in enumerate(_ordered_turn_metrics(per_turn)):
        label = f"Turn {i + 1}"
        for stp in (turn.get("context_steps") or []):
            tokens = stp.get("input_tokens")
            if tokens is None:
                continue
            point = {
                "x": len(points) + 1,
                "turn": label,
                "step": stp.get("step"),
                "input_tokens": tokens,
                "output_tokens": stp.get("output_tokens"),
                # The final call of a turn requests no tool (finish_reason
                # "stop"); naming it the answer keeps the hover honest.
                "tool": stp.get("tool") or "final answer",
                "llm_s": stp.get("llm_s"),
                "delta": (tokens - prev["input_tokens"]) if prev else None,
                "prev_label": (f"{prev['turn']} - step {prev['step']}"
                               if prev else None),
                "new_turn": bool(prev) and prev["turn"] != label,
            }
            points.append(point)
            prev = point
    return points


def _context_hover(point):
    """Hover text for one point on the continuous context curve.

    Carries everything the old on-plot label said and more: the turn-local step
    number, so a point sitting at session step 4 still reads as "Turn 2 - step
    1", plus the tool that call asked for -- the rise to the next point is that
    tool's result landing in the prompt.
    """
    delta = ""
    if point["delta"] is not None:
        across = " (carried over from the previous turn)" if point["new_turn"] else ""
        delta = (f"{point['delta']:+,} tok since {point['prev_label']}"
                 f"{across}<br>")
    out = point["output_tokens"]
    return (
        f"<b>{point['turn']} - step {point['step']}</b> "
        f"(session step {point['x']})<br>"
        f"{point['input_tokens']:,} input tokens<br>{delta}"
        f"requested: <b>{point['tool']}</b><br>"
        f"LLM call: {point['llm_s']}s, "
        f"{out if out is not None else 'n/a'} output tokens"
    )


def build_context_growth_figure(points, window=None) -> go.Figure:
    """One continuous input-token curve for the whole session.

    Turns stay separate traces so the legend can isolate or hide one, but each
    turn is bridged to the next by a dotted connector in the incoming turn's
    color. The old chart restarted every turn at x=1, which drew the session's
    context climb as a set of disconnected short lines and hid the one thing the
    chart exists to show.

    Point labels are deliberately absent. One tool name per marker meant a
    session-length axis carrying dozens of overlapping annotations; the name now
    lives in the hover box, where it stays readable however long the session runs.

    `window` caps how many steps are visible, so a long session shows a moving
    window over its newest steps instead of compressing everything into the
    panel width. Drag on the plot to pan back through the rest; there is no
    range slider under the axis -- the mini-map read as a second chart and cost
    vertical space the curve itself wanted.
    """
    fig = go.Figure()
    if not points:
        return fig

    # Group consecutive points by turn. They are already consecutive by
    # construction, so this also survives a turn label repeating (it cannot
    # today, but a run of points is the thing being drawn, not the label).
    groups = []
    for point in points:
        if groups and groups[-1]["label"] == point["turn"]:
            groups[-1]["points"].append(point)
        else:
            groups.append({"label": point["turn"], "points": [point]})

    # Beyond this many turns the band labels collide into unreadable mush; the
    # legend and the hover box still name every turn.
    label_bands = len(groups) <= 12

    for gi, group in enumerate(groups):
        color = TURN_COLORS[gi % len(TURN_COLORS)]
        pts = group["points"]
        # A faint band per turn, labelled once. This is what replaces the
        # per-point tool names: turn boundaries stay visible at a glance without
        # writing anything on the curve itself.
        fig.add_vrect(
            x0=pts[0]["x"] - 0.5, x1=pts[-1]["x"] + 0.5,
            fillcolor=color, opacity=0.05, line_width=0, layer="below",
            annotation_text=(group["label"] if label_bands else ""),
            annotation_position="top left",
            annotation_font=dict(size=10, color=SUBINK),
        )
        if gi:
            # The bridge across the turn boundary. Dotted so it still reads as a
            # new turn starting, drawn without hover so the two points it joins
            # keep their own hover boxes.
            prev_pt = groups[gi - 1]["points"][-1]
            fig.add_trace(go.Scatter(
                x=[prev_pt["x"], pts[0]["x"]],
                y=[prev_pt["input_tokens"], pts[0]["input_tokens"]],
                mode="lines", line=dict(width=2, color=color, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in pts],
            y=[p["input_tokens"] for p in pts],
            name=group["label"], mode="lines+markers",
            line=dict(width=2, color=color), marker=dict(size=8, color=color),
            hovertext=[_context_hover(p) for p in pts], hoverinfo="text",
        ))

    n = len(points)
    lo, hi = 0.5, n + 0.5
    if window and n > window:
        lo = n - window + 0.5
    # Integer ticks whatever the length: a step axis has no half steps, and
    # Plotly's autoticks happily produce 2.5 on a short session.
    dtick = max(1, -(-n // 25))
    fig.update_layout(
        title="Input tokens at each agent step (whole session)",
        xaxis=dict(
            title="Agent step(LLM Call+Tool Use)",
            dtick=dtick, range=[lo, hi],
        ),
        yaxis=dict(title="Input tokens"),
        height=430,
    )
    if lo > 0.5:
        # With an explicit x range Plotly still autoscales y over ALL points, so
        # a moving window would sit squashed against the top of the panel. Fit y
        # to what is actually in view instead.
        ys = [p["input_tokens"] for p in points if lo <= p["x"] <= hi]
        if ys:
            pad = max(1.0, (max(ys) - min(ys)) * 0.15)
            fig.update_layout(yaxis=dict(title="Input tokens",
                                         range=[min(ys) - pad, max(ys) + pad]))
    return fig


# ---------------------------------------------------------------------------
# Live trace polling (for the continuous curve)
# ---------------------------------------------------------------------------

def _trace_id_of(trace) -> str:
    """Trace id from a full-trace dict, whichever key this MLflow build used."""
    info = trace.get("info") if isinstance(trace, dict) else None
    if isinstance(info, dict):
        for key in ("trace_id", "request_id"):
            if info.get(key):
                return str(info[key])
    for span in _trace_spans(trace):
        if isinstance(span, dict) and span.get("trace_id"):
            return str(span["trace_id"])
    return ""


def _download_trace(trace_id: str):
    """One full trace as a plain dict, or None if it cannot be fetched.

    The same conversion fetch_full_traces does, minus the caching: the live poll
    downloads each trace exactly once and keeps it in session_state, so a second
    cache layer keyed on the id would only duplicate the memory.
    """
    try:
        tr = mlflow.get_trace(trace_id)
    except Exception:
        return None
    for convert in (lambda t: json.loads(t.to_json()), lambda t: t.to_dict()):
        try:
            return convert(tr)
        except Exception:
            continue
    return None


def poll_live_traces(tracking_uri: str, session_id: str, experiment_id=None):
    """Re-scan MLflow for this session's traces; return (traces, error).

    Called on a timer by the Context & tools tab's live curve, so it is
    incremental: one search_traces call to see which trace ids exist now, then a
    download of only the ids not already in hand. A poll on a session that has
    not advanced costs the search and nothing else.

    Traces accumulate in session_state, seeded from whatever Load already
    fetched, and are written back to the session's traces.json -- the file the
    rest of the dashboard renders from -- whenever a new turn lands, so the live
    curve and the other tabs cannot drift apart by more than one rerun.

    Note that a turn's spans reach MLflow when that TURN ends, not when each
    step ends: the curve extends a turn at a time while the agent works, which
    is as live as this telemetry gets.

    (Uses ordered_turns, defined below in the turn-ordering section -- a
    module-level name resolved when this runs, long after import.)
    """
    if not session_id:
        return [], "no session id"

    store = st.session_state.setdefault("live_traces", {})
    by_id = store.get(session_id)
    if by_id is None:
        by_id = {}
        for tr in ((st.session_state.get("loaded") or {}).get("full_traces") or []):
            tid = _trace_id_of(tr)
            if tid:
                by_id[tid] = tr
        store[session_id] = by_id

    mlflow.set_tracking_uri(tracking_uri)
    # The summary fetch is @st.cache_data; without clearing it, every poll would
    # be answered from the first poll's cached trace-id list and the curve would
    # never grow.
    try:
        fetch_session_traces.clear()
    except Exception:
        pass
    try:
        summary, err = fetch_session_traces(tracking_uri, session_id, experiment_id)
    except Exception as e:
        return list(by_id.values()), f"MLflow poll failed: {e}"
    if err:
        return list(by_id.values()), err

    ids = []
    if not summary.empty and "trace_id" in summary.columns:
        ids = [str(t) for t in summary["trace_id"].dropna().tolist()]
    fetched = 0
    for tid in ids:
        if tid in by_id:
            continue
        trace = _download_trace(tid)
        if trace is not None:
            by_id[tid] = trace
            fetched += 1

    traces = [tr for _, tr in ordered_turns(list(by_id.values()))]
    if fetched:
        save_traces_json(session_id, traces)
    return traces, ""


# ---------------------------------------------------------------------------
# Turn ordering
# ---------------------------------------------------------------------------
# All that remains of the Graphviz flow-diagram section. The diagram was
# removed from the Analysis tab, and with it _safe_id / _dot_escape /
# _dot_header / _DOT_TYPE_STYLE / _turn_dot_lines / build_turn_flowchart /
# build_session_flowchart, none of which had another caller. ordered_turns
# stays: poll_live_traces uses it to merge freshly polled traces back into
# turn order.

def ordered_turns(traces):
    """Return [(turn_label, trace), ...] ordered by hermes.turn.number then start.

    turn_label is the printable turn number (a plain string); traces with no turn
    attribute fall back to their 1-based position.
    """
    def _key(tr):
        t = _turn_of_trace(tr)
        if isinstance(t, int):
            return (0, t)
        ns = [s["start"] for s in _norm_spans(tr) if s["start"] is not None]
        return (1, min(ns) if ns else 0)

    out = []
    for ti, tr in enumerate(sorted(traces or [], key=_key)):
        t = _turn_of_trace(tr)
        try:
            lbl = str(int(float(t)))
        except (TypeError, ValueError):
            lbl = str(t) if t is not None else str(ti + 1)
        out.append((lbl, tr))
    return out


def session_overview_figure(session_id=None, ip="127.0.0.1", port="5004",
                            prom_url=None, verbose=True):
    """Build the Overview span-waterfall + CPU/GPU figure for one session, headless.

    Mirrors the dashboard's Load handler without any Streamlit UI: resolve the
    session's experiment, fetch its full traces, pull per-turn CPU/GPU from
    Prometheus, then hand it all to build_session_waterfall_figure. Importing
    this module does NOT launch the dashboard (the UI below is guarded by
    _running_under_streamlit), so a notebook can call this directly:

        import hermes_profiler
        hermes_profiler.session_overview_figure().show()

    session_id=None uses the newest session on the MLflow server, exactly the
    "Fetch -> newest first" the dashboard's Fetch button does. Returns a Plotly
    Figure; call .show() on it to render inline.
    """
    prom_url = prom_url or os.environ.get("PROM_URL", "http://127.0.0.1:9090")
    uri = build_tracking_uri(ip, port)

    if not session_id:
        ids = fetch_session_ids(uri)
        if not ids:
            raise RuntimeError(f"No sessions found on the MLflow server at {uri}.")
        session_id = ids[0]
    if verbose:
        print(f"Session: {session_id}")

    info = resolve_run(uri, session_id)
    if not info:
        raise RuntimeError(f"No traces found for session_id={session_id!r} at {uri}.")

    summary_df, _err = fetch_session_traces(uri, session_id, info.get("experiment_id"))
    trace_ids = (
        tuple(str(t) for t in summary_df["trace_id"].dropna())
        if not summary_df.empty and "trace_id" in summary_df.columns else ()
    )
    full_traces, _ = fetch_full_traces(uri, trace_ids)

    save_traces_json(session_id, full_traces)
    full_traces = load_traces_json(session_id)

    local_dir = os.path.join(PROFILING_CACHE_DIR, session_id, ARTIFACT_DIR)
    save_session_cpu_gpu(local_dir, prom_url, full_traces)

    cpu_df = parse_timestamps(read_csv(os.path.join(local_dir, "cpu_hermes_trace.csv")))
    gpu_df = parse_timestamps(read_csv(os.path.join(local_dir, "gpu_system_wide.csv")))
    tool_df = parse_timestamps(read_csv(os.path.join(local_dir, "tool_execution.csv")))

    fig = build_session_waterfall_figure(full_traces, cpu_df, gpu_df, tool_df)
    # The span names are the row-1 y tick labels. The dashboard's tight left
    # margin (l=10) is fine in its wide layout but clips them in a notebook.
    # Size the left margin to the longest (monospace) label, ~6.5 px/char, and
    # also turn on automargin so Plotly can widen it further if needed.
    labels = fig.layout.yaxis.ticktext or ()
    maxlen = max((len(str(t)) for t in labels), default=12)
    m = fig.layout.margin
    fig.update_layout(margin=dict(l=min(360, max(90, int(maxlen * 6.5) + 24)),
                                  r=m.r, t=m.t, b=m.b))
    fig.update_yaxes(automargin=True, row=1, col=1)
    return fig


def session_detail_links(session_id=None, ip="127.0.0.1", port="5004",
                         dashboard_port=8501, host=None,
                         proxy_base="https://notebooks.amd.com"):
    """Markdown with two 'detailed view' links for a session:

      1. the Streamlit telemetry dashboard (port dashboard_port), and
      2. the MLflow UI trace view for this session's experiment (port port).

    By default builds AMD hosted-notebook proxy links, matching
    video_generation_workshop.ipynb:

        {proxy_base}/{hostname}/proxy/{service_port}/

    where hostname is socket.gethostname(). The base is one knob (argument, or the
    HERMES_PROXY_BASE env var which wins) interpreted two ways:

        "https://notebooks.amd.com"  (has "://")  -> proxy: {base}/<hostname>/proxy/<port>/
        ""                            (empty)      -> direct: http://127.0.0.1:<port>/
        "10.0.0.5"                    (bare host)  -> direct: http://10.0.0.5:<port>/

    On the AMD hosted platform a direct http://host:port link does not work
    (127.0.0.1 / the container IP is not reachable from the browser), so the proxy
    path is the default; use "" when the browser is on the same machine or you
    SSH-forward the port.

    The MLflow link points at the experiment's Traces PAGE in the MLflow UI, not
    the OTLP ingest endpoint (/v1/traces), which is not browsable.
    """
    # The HERMES_PROXY_BASE env var, when set, overrides the argument so the base
    # can be chosen once (in one cell or the shell) for every call.
    env_base = os.environ.get("HERMES_PROXY_BASE")
    if env_base is not None:
        proxy_base = env_base

    uri = build_tracking_uri(ip, port)
    if not session_id:
        ids = fetch_session_ids(uri)
        session_id = ids[0] if ids else None
    info = resolve_run(uri, session_id) if session_id else None
    exp_id = info.get("experiment_id") if info else None
    frag = f"#/experiments/{exp_id}/traces" if exp_id else ""

    # One knob, two shapes:
    #   value WITH "://"  -> a proxy base; build {base}/<hostname>/proxy/<port>/
    #   value WITHOUT it   -> a direct host ("" means 127.0.0.1); build http://<host>:<port>/
    if proxy_base and "://" in proxy_base:
        base = proxy_base.rstrip("/")
        hostname = socket.gethostname()
        dash_url = f"{base}/{hostname}/proxy/{dashboard_port}/"
        mlflow_url = f"{base}/{hostname}/proxy/{port}/{frag}"
        where = f"the notebook proxy (`{base}/{hostname}/proxy/<port>/`)"
    else:
        h = (proxy_base or "").strip() or host or ip
        dash_url = f"http://{h}:{dashboard_port}/"
        mlflow_url = f"http://{h}:{port}/{frag}"
        where = f"`http://{h}:<port>/` directly (no proxy)"

    return (
        f"**Detailed views for session `{session_id}`:**\n\n"
        f"1. [Open the telemetry dashboard (Streamlit, port {dashboard_port})]({dash_url}) "
        f"— then click **Fetch → Load** and select this session.\n"
        f"2. [Open the MLflow trace view (port {port})]({mlflow_url})\n\n"
        f"<sub>Links use {where}.</sub>"
    )


def show_session_overview(session_id=None, ip="127.0.0.1", port="5004",
                          prom_url=None, dashboard_port=8501, host=None,
                          proxy_base="https://notebooks.amd.com"):
    """Notebook helper: render the Overview graph, then the two detail links.

    Resolves the session id ONCE (newest when session_id is None) so the graph
    and the links describe the SAME run. Links default to the AMD notebook proxy
    (see session_detail_links); pass proxy_base=None for direct host:port links.
    IPython is imported lazily so importing this module elsewhere never needs it.
    """
    import logging
    for _n in list(logging.root.manager.loggerDict):
        if _n.startswith("streamlit"):
            logging.getLogger(_n).setLevel(logging.ERROR)
    from IPython.display import display, Markdown

    uri = build_tracking_uri(ip, port)
    if not session_id:
        ids = fetch_session_ids(uri)
        if not ids:
            raise RuntimeError(f"No sessions found on the MLflow server at {uri}.")
        session_id = ids[0]

    fig = session_overview_figure(session_id, ip=ip, port=port, prom_url=prom_url,
                                  verbose=False)
    fig.show()
    display(Markdown(session_detail_links(session_id, ip=ip, port=port,
                                          dashboard_port=dashboard_port, host=host,
                                          proxy_base=proxy_base)))


# Preset choices for the link-base dropdown. Each is (label, value); the value is
# what session_detail_links interprets (has "://" -> proxy, else a direct host,
# "" -> 127.0.0.1). "__custom__" reveals a free-text box so any base can be typed.
_PROXY_PRESETS = [
    ("AMD hosted proxy (notebooks.amd.com)", "https://notebooks.amd.com"),
    ("Local - 127.0.0.1 direct links", ""),
    ("Custom (type below)…", "__custom__"),
]


def overview_selector(session_id=None, ip="127.0.0.1", port="5004", prom_url=None,
                      dashboard_port=8501):
    """Notebook UI: an editable dropdown to choose where the links point, then the
    Overview graph + detail links, re-rendered whenever you click the button.

    The dropdown defaults to the AMD hosted proxy and offers 127.0.0.1 plus a
    Custom option with a text box (type anything: "" for 127.0.0.1, a bare host
    like "10.0.0.5", or a proxy URL like "https://my-proxy"). The choice is stored
    in HERMES_PROXY_BASE, so the other overview cells pick it up too.

    Requires ipywidgets. If it is not installed, this transparently falls back to
    show_session_overview() using whatever HERMES_PROXY_BASE / default is set.
    """
    try:
        import ipywidgets as W
    except Exception:
        print("(ipywidgets not installed — showing the overview with the current "
              "HERMES_PROXY_BASE; `pip install ipywidgets` to get the dropdown.)")
        return show_session_overview(session_id, ip=ip, port=port, prom_url=prom_url,
                                     dashboard_port=dashboard_port)
    from IPython.display import display

    current = os.environ.get("HERMES_PROXY_BASE", "https://notebooks.amd.com")
    preset_values = [v for _, v in _PROXY_PRESETS if v != "__custom__"]
    initial = current if current in preset_values else "__custom__"

    dd = W.Dropdown(options=_PROXY_PRESETS, value=initial, description="Link base:",
                    style={"description_width": "initial"},
                    layout=W.Layout(width="520px"))
    custom = W.Text(value=("" if current in preset_values else current),
                    placeholder='"" = 127.0.0.1  |  "10.0.0.5"  |  "https://my-proxy"',
                    description="Custom:", style={"description_width": "initial"},
                    layout=W.Layout(width="520px"))
    custom.layout.display = "" if initial == "__custom__" else "none"
    btn = W.Button(description="Show overview", button_style="primary")
    out = W.Output()

    def _on_dd(_):
        custom.layout.display = "" if dd.value == "__custom__" else "none"

    def _base():
        return custom.value.strip() if dd.value == "__custom__" else dd.value

    def _render(_):
        os.environ["HERMES_PROXY_BASE"] = _base()
        with out:
            out.clear_output(wait=True)
            show_session_overview(session_id, ip=ip, port=port, prom_url=prom_url,
                                  dashboard_port=dashboard_port)

    dd.observe(_on_dd, names="value")
    btn.on_click(_render)
    display(W.VBox([dd, custom, btn]), out)
    _render(None)   # render once with the initial selection


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _running_under_streamlit():
    """True only when launched via `streamlit run`, not on a plain import.

    streamlit.runtime.exists() is True only inside a live Streamlit runtime, so
    importing this module from a notebook kernel skips the dashboard UI below and
    exposes just the functions above. (A Streamlit server running in another
    process does not count: exists() is per-process.) Falls back to the __main__
    check on old Streamlit builds that predate the runtime API.
    """
    try:
        from streamlit.runtime import exists as _st_runtime_exists
        return _st_runtime_exists()
    except Exception:
        return __name__ == "__main__"


if _running_under_streamlit():
    # A bare hostname like "0" (common inside a container) is noise, so the chip
    # only appears when the host name carries real information.
    _host = socket.gethostname()
    host_chip = (
        f'<span class="amd-chip">{_host}</span>' if len(_host) > 2 else ""
    )

    st.markdown(
        f"""
    <div class="amd-hero">
      <div class="amd-hero-title">Hermes <span class="accent">Telemetry</span> Dashboard</div>
      <p class="amd-hero-sub">
        Agentic AI profiling on AMD Instinct. Correlate CPU and GPU utilization with
        tool-execution spans, inspect per-turn traces, and analyze tool usage for a
        recorded Hermes session.
      </p>
      <div class="amd-chip-row">
        <span class="amd-chip live"><span class="dot"></span>MLflow telemetry</span>
        <span class="amd-chip">AMD Instinct MI300X</span>
        <span class="amd-chip">ROCm</span>
        {host_chip}
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        # AMD branding + telemetry header. Machine name is the host running Streamlit.
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=132)
        st.markdown("### Agent Telemetry")
        st.caption(f"{socket.gethostname()} | Hermes Orchestration")
        st.divider()

        st.header("Connection")
        ip = st.text_input("MLflow server IP / host", value="127.0.0.1")
        port = st.text_input("MLflow server port", value="5004")
        prom_url = st.text_input(
            "Prometheus URL (CPU/GPU metrics)",
            value=os.environ.get("PROM_URL", "http://127.0.0.1:9090"),
            help="Where hermes-otel's host-metrics (CPU/GPU) land as OTel metrics "
                 "(e.g. the Grafana LGTM backend). Traces still come from the "
                 "MLflow server above.",
        )

        # Session ID selector with a Fetch button to its right. The button column is
        # handled first in code so a fetch updates the list before the selectbox below
        # reads it (same run, no extra rerun). columns render left→right regardless of
        # code order, so the button still sits to the right of the box.
        # 3:1 clipped the Fetch label to "Fetc" in the narrow sidebar; 2:1 plus
        # width="stretch" on the button keeps the word intact.
        c_sel, c_btn = st.columns([2, 1], vertical_alignment="bottom")
        with c_btn:
            do_fetch = st.button(
                "Fetch", width="stretch",
                help="List all session IDs on this MLflow server.",
            )
        if do_fetch:
            _uri = build_tracking_uri(ip, port)
            with st.spinner("Fetching session IDs …"):
                try:
                    fetch_session_ids.clear()
                    st.session_state["session_ids"] = fetch_session_ids(_uri)
                    st.session_state["session_ids_uri"] = _uri
                except Exception as e:
                    st.session_state["session_ids"] = []
                    st.error(f"Could not fetch sessions from {_uri}: {e}")

        _sids = st.session_state.get("session_ids", [])
        with c_sel:
            # One box that is both a dropdown and a text field: pick a fetched session
            # or type any session id (accept_new_options makes the selectbox editable).
            session_id = st.selectbox(
                "Session ID",
                options=_sids,
                index=None,
                accept_new_options=True,
                placeholder="Select a fetched session or type a Session ID…",
                help="Pick a session on the server, or type any session ID. "
                     "Click Fetch to populate the list.",
            ) or ""
        if not _sids:
            st.caption("Click **Fetch** to list session IDs, or just type one in above.")

        load = st.button("Load / Reload", type="primary", width="stretch",
                         help="Fetch this session's latest run. Click again after "
                              "running more queries to pull the new data.")

    # On Load: fetch everything and stash it in session_state. All rendering below
    # reads from session_state, so later widget clicks (radio, download, tab switch)
    # rerun the script without re-triggering Load or resetting the view.
    if load:
        if not session_id.strip():
            st.warning("Please enter a session ID.")
            st.stop()

        # Fetch everything fresh: drop the @st.cache_data caches BEFORE fetching so a
        # resumed session's newly-added turns are pulled (and rewritten to the cache
        # file) on the first Load click, not the second.
        for _cache in (resolve_run, download_profiling, fetch_session_traces, fetch_full_traces):
            try:
                _cache.clear()
            except Exception:
                pass

        tracking_uri = build_tracking_uri(ip, port)
        with st.spinner("Resolving session → experiment …"):
            try:
                info = resolve_run(tracking_uri, session_id.strip())
            except Exception as e:
                st.error(f"Could not reach MLflow at {tracking_uri}: {e}")
                st.stop()
        if not info:
            st.error(f"No traces found with session_id = '{session_id}'.")
            st.stop()

        # Pre-fetch this session's full traces (JSON with all spans) up front: they
        # drive the Tab-1 timeline waterfall / Analysis tab AND, now, the per-turn
        # windows used to pull this session's CPU/GPU from Prometheus below. There
        # is no more profiling/ MLflow artifact to download for CPU/GPU - hermes-otel
        # exports them as OTel metrics, not files (see fetch_session_cpu_gpu).
        full_traces = None
        turn_count = 0
        total_latency_ms = 0.0
        with st.spinner("Fetching session traces …"):
            try:
                _summary_df, _summary_err = fetch_session_traces(
                    tracking_uri, session_id.strip(), info.get("experiment_id")
                )
                if not _summary_err and not _summary_df.empty:
                    turn_count = len(_summary_df)
                    total_latency_ms = _summary_df.attrs.get("total_latency_ms", 0.0)
                    if "trace_id" in _summary_df.columns:
                        _tids = tuple(str(t) for t in _summary_df["trace_id"].dropna().tolist())
                        full_traces, _ = fetch_full_traces(tracking_uri, _tids)
            except Exception:
                full_traces = None

        # Overwrite the profiling-cache file with the freshly-fetched traces (so a
        # resumed session's later Load reflects ALL its turns), then read them back:
        # the on-disk file under profiling_cache is the ONLY source the dashboard
        # renders from - no in-memory fallback.
        _sid = session_id.strip()
        if full_traces:
            save_traces_json(_sid, full_traces)
        full_traces = load_traces_json(_sid)
        if full_traces and not turn_count:
            turn_count = len(full_traces)

        # For each turn (trace) in this session, fetch its CPU/GPU from Prometheus
        # over that turn's own [start, end] window and merge every turn into one
        # session-level CSV - same technique as fetch_session_metrics.py, just run
        # per-turn instead of once for the whole session, so long idle gaps between
        # turns never blow past Prometheus's per-query point limit. tool_execution.csv
        # is rebuilt straight from the traces' own hermes.tool.* span attributes
        # (no Prometheus query needed for that one).
        local_dir = os.path.join(PROFILING_CACHE_DIR, _sid, ARTIFACT_DIR)
        with st.spinner("Fetching CPU/GPU from Prometheus, per turn …"):
            try:
                _stats = save_session_cpu_gpu(local_dir, prom_url, full_traces)
                if not full_traces:
                    st.warning("No traces to derive turn windows from — CPU/GPU will be empty.")
                elif _stats["cpu_hermes_points"] == 0 and _stats["gpu_points"] == 0:
                    st.warning(
                        "No CPU/GPU samples found for this session's turn windows in "
                        f"Prometheus at {prom_url}. This session may predate metrics "
                        "being enabled (flush_interval_ms/metrics backend), or the "
                        "Prometheus URL doesn't point at the right server."
                    )
            except Exception as e:
                st.warning(f"Could not fetch CPU/GPU from Prometheus at {prom_url}: {e}")

        st.session_state["loaded"] = {
            "tracking_uri": tracking_uri,
            "info": info,
            "session_id": session_id.strip(),
            "local_dir": local_dir,
            "full_traces": full_traces,
            "turn_count": turn_count,
            "total_latency_ms": total_latency_ms,
            "cpu_df": parse_timestamps(read_csv(os.path.join(local_dir, "cpu_hermes_trace.csv"))),
            "gpu_df": parse_timestamps(read_csv(os.path.join(local_dir, "gpu_system_wide.csv"))),
            "tool_df": parse_timestamps(read_csv(os.path.join(local_dir, "tool_execution.csv"))),
            "metrics": compute_session_metrics(full_traces) if full_traces else None,
        }
        # Loading a session invalidates per-session tab state from any previous one -
        # clear cached traces & analysis so those tabs don't show stale data. (The
        # @st.cache_data fetch caches are already cleared at the top of this handler.)
        for _k in ("traces_result", "analysis_result", "analysis_run"):
            st.session_state.pop(_k, None)

    # Nothing loaded yet → prompt and stop.
    # Require a fresh Load if nothing is cached, or if the cache predates the current
    # schema (missing newer keys like local_dir/session_id from an older run).
    _data = st.session_state.get("loaded")
    _cached_metrics = (_data or {}).get("metrics")
    # A cached dict from an older METRICS_SCHEMA is stale even though its key
    # exists; reading it with newer render code raises KeyError on a key that did
    # not exist yet, so force a fresh Load instead.
    _metrics_stale = bool(_cached_metrics) and _cached_metrics.get("schema") != METRICS_SCHEMA
    if (not _data or "local_dir" not in _data or "session_id" not in _data
            or "metrics" not in _data or _metrics_stale):
        st.info("Enter the MLflow server IP, port, and a session ID in the sidebar, then click **Load**.")
        st.stop()

    # Pull the persisted data (survives radio/download/tab interactions).
    tracking_uri = _data["tracking_uri"]
    info = _data["info"]
    cpu_df = _data["cpu_df"]
    gpu_df = _data["gpu_df"]
    tool_df = _data["tool_df"]
    local_dir = _data["local_dir"]
    sess_id = _data["session_id"]
    loaded_full_traces = _data.get("full_traces")
    turn_count = _data.get("turn_count", 0)
    total_latency_ms = _data.get("total_latency_ms", 0.0)
    metrics = _data.get("metrics")

    st.write(f"**Tracking URI:** `{tracking_uri}`")
    st.success(f"Found session `{sess_id}` in experiment `{info['experiment']}`.")

    # Session-wide summary strip: total turns and their combined latency (the sum of
    # each turn's MLflow trace execution_duration, computed once in
    # fetch_session_traces and carried through from the Load click). Each metric sits
    # in its own bordered card - a bare st.metric has no visual separation from the
    # page background, so this reads more like a dashboard KPI row. The metric value's
    # font size is normalized in the app-wide style block near set_page_config so it
    # stays inside the card.
    m1, m2, m3 = st.columns(3)
    with m1:
        with st.container(border=True):
            st.metric("Session ID", sess_id)
    with m2:
        with st.container(border=True):
            st.metric("Turns", turn_count or "n/a")
    with m3:
        with st.container(border=True):
            st.metric(
                "Total Latency",
                format_latency_ms(total_latency_ms) if turn_count else "n/a",
            )

    if cpu_df.empty and gpu_df.empty:
        st.warning(
            "No CPU/GPU timeline data found for this session's turns in Prometheus. "
            "This session may predate metrics being enabled, or the Prometheus URL "
            "in the sidebar doesn't point at the right server."
        )
        st.stop()

    (tab_overview, tab_separate, tab_context_tools, tab_traces,
     tab_analysis) = st.tabs(
        ["Overview", "CPU / GPU separate", "Context & tools", "Traces", "Analysis"]
    )

    with tab_overview:
        # Toggle: ON shows the full-session span waterfall correlated with CPU/GPU on
        # a shared wall-clock axis; OFF shows the standalone per-session CPU/GPU chart.
        show_timeline = st.toggle(
            "Show trace timeline (full-session span waterfall)",
            value=False,
            help="Render every turn's spans on one absolute wall-clock axis, stacked "
                 "over the CPU/GPU timeline so the two correlate directly. Idle time "
                 "between prompts appears as the same gap in both. Traces are fetched "
                 "when you click Load.",
        )
        if show_timeline:
            if not loaded_full_traces:
                st.info(
                    "No traces were fetched for this session. Re-click **Load** (the "
                    "traces are pulled then), or confirm MLflow tracing is enabled."
                )
            else:
                st.plotly_chart(
                    build_session_waterfall_figure(loaded_full_traces, cpu_df, gpu_df, tool_df),
                    width="stretch",
                )
                # If the waterfall looks empty/misaligned, the span field names in this
                # MLflow build may differ - inspect the raw traces here to confirm.
                with st.expander("Debug: raw trace JSON (all turns)", expanded=False):
                    # loaded_full_traces was read from this file at Load.
                    st.caption(f"Read from `{traces_cache_path(sess_id)}`")
                    st.download_button(
                        "⬇ Download traces.json",
                        data=json.dumps(loaded_full_traces, indent=2, default=str),
                        file_name="traces.json", mime="application/json",
                        key="dl_traces_json",
                    )
                    st.json(loaded_full_traces)
        else:
            st.plotly_chart(build_figure(cpu_df, gpu_df, tool_df), width="stretch")

        with st.expander("Tool breakdown table", expanded=True):
            if tool_df.empty:
                st.write("No tool_execution.csv data.")
            else:
                # Show the row number starting at 1 instead of the 0-based index.
                _disp = tool_df.copy()
                _disp.index = range(1, len(_disp) + 1)
                show_left_table(_disp)

    with tab_separate:
        sources = {
            "CPU Usage": ("cpu_hermes_trace.csv", cpu_df),
            "GPU usage": ("gpu_system_wide.csv", gpu_df),
            "Tool Track": ("tool_execution.csv", tool_df),
        }

        show_panel = st.toggle("Show CSV panel (compare live)", value=False,
                               help="Open a side panel with the raw CSV next to the graphs.")

        def _render_graphs():
            st.subheader("CPU utilization")
            st.plotly_chart(
                build_single_figure(cpu_df, "cpu_pct", "CPU %", BLUE, tool_df=tool_df,
                                     y_range=[0, 102],
                                     y_title="CPU % (hermes + children)"),
                width="stretch",
            )
            st.subheader("GPU utilization")
            st.plotly_chart(
                build_single_figure(gpu_df, "gfx_busy_pct", "GPU %", AMD_RED,
                                     y_range=[0, 102], tool_df=tool_df),
                width="stretch",
            )

        def _render_csv_panel():
            st.markdown("#### Raw data")
            choice = st.radio("View data", list(sources.keys()),
                              label_visibility="collapsed", horizontal=True)
            fname, df = sources[choice]
            st.caption(f"`{fname}`")
            if df is None or df.empty:
                st.info(f"No data in {fname}.")
            else:
                show_left_table(df, height=430)
                st.download_button(
                    label=f"⬇ Download {fname}",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=fname, mime="text/csv", key=f"dl_{fname}",
                )

        if show_panel:
            # Split view: graphs on the left, CSV panel on the right. Each column is
            # a fixed-height scrollable container so they stay top-aligned and scroll
            # independently (otherwise the two stacked graphs push the GPU chart far
            # below the CSV panel).
            left, right = st.columns([3, 2], gap="large")
            with left:
                with st.container(height=620):
                    _render_graphs()
            with right:
                with st.container(height=620):
                    _render_csv_panel()
        else:
            _render_graphs()

    with tab_context_tools:
        st.subheader("Context & tools")
        st.caption(
            "Derived from this session's span trees. Nothing here needs new "
            "instrumentation -- it is arithmetic over telemetry the patched "
            "hermes-otel plugin already emits."
        )

        def _fmt(value, suffix="", nd=1):
            """Render a metric, distinguishing 'not applicable' from zero.

        A turn that called no tools has no time-to-first-tool; showing 0 there
        would read as 'instant' rather than 'never happened'.
        """
            if value is None:
                return "n/a"
            if isinstance(value, float):
                return f"{value:.{nd}f}{suffix}"
            return f"{value}{suffix}"

        if not metrics or not metrics.get("per_turn", []):
            st.info(
                "No traces loaded for this session, so these metrics cannot be "
                "derived. Click **Load / Reload** in the sidebar."
            )
        else:
            per_turn = metrics.get("per_turn", [])
            n_turns = metrics.get("turns", 0)

            # Each card aggregates differently -- total, mean, mean, pooled -- so
            # each says which it is. A bare mean hides the worst turn, which is the
            # one worth investigating, so the extremum is named alongside.
            section("Context growth per step")
            # One continuous curve for the whole session: input tokens against a
            # step index that keeps counting across turns, with consecutive turns
            # bridged. The intercept is the fixed prompt cost (system message + tool
            # schemas); the slope is what the agent adds to its own context as it
            # works -- across the session, not just inside one turn.
            #
            # These two sit OUTSIDE the fragment because they set its poll timer,
            # and run_every is fixed when the fragment is declared -- changing them
            # has to re-run the page for the new timer to take effect. Everything
            # that does not touch the timer lives inside the fragment instead.
            _c_live, _c_every, _ = st.columns([1.2, 1, 2])
            with _c_live:
                ctx_live = st.toggle(
                    "Live follow", value=False, key="ctx_live",
                    help="Re-poll MLflow for this session and extend the curve as "
                         "the agent works. A turn's spans reach MLflow when that "
                         "turn ends, so the curve grows a turn at a time.",
                )
            with _c_every:
                ctx_every = st.number_input(
                    "Refresh (s)", min_value=2, max_value=60, value=5, step=1,
                    key="ctx_every", disabled=not ctx_live,
                    help="How often to poll while Live follow is on.",
                )

            # The chart lives in a fragment so Live follow reruns ONLY the chart on
            # its timer. A page-level rerun would also reset st.tabs() back to
            # Overview every few seconds (the same trap the Analysis poller
            # documents), which would make live mode unusable.
            @st.fragment(run_every=(int(ctx_every) if ctx_live else None))
            def _render_context_growth():
                # Inside the fragment: moving the window redraws the chart alone,
                # without re-running the page.
                _c_window, _ = st.columns([1.6, 2.4])
                with _c_window:
                    ctx_window = st.slider(
                        "Steps in view", min_value=10, max_value=200, value=40,
                        step=5, key="ctx_window",
                        help="Width of the moving window. Once the session has more "
                             "steps than this the chart follows the newest ones; "
                             "drag on the chart itself to pan back to earlier steps.",
                    )

                turns_now, poll_err = per_turn, ""
                if ctx_live:
                    traces_now, poll_err = poll_live_traces(
                        tracking_uri, sess_id, info.get("experiment_id"))
                    if traces_now:
                        fresh = compute_session_metrics(traces_now)
                        turns_now = fresh.get("per_turn", per_turn)
                        # Keep the rest of the page in step: the tables below this
                        # chart read `metrics` off session_state, so they catch up
                        # on the next rerun instead of contradicting the curve.
                        _data["full_traces"] = traces_now
                        _data["metrics"] = fresh
                        _data["turn_count"] = fresh.get("turns", 0)

                points = context_growth_points(turns_now)
                if len(points) < 2:
                    st.info(
                        "Context growth needs at least two LLM calls in the "
                        "session; only one has been recorded so far."
                    )
                else:
                    st.plotly_chart(
                        style_figure(build_context_growth_figure(
                            points, window=int(ctx_window))),
                        width="stretch",
                    )
                    turn_n = len({p["turn"] for p in points})
                    # Only describe the turn bridges when there is more than one
                    # turn to bridge.
                    across = (
                        "The x axis runs continuously across turns: turn 2's first "
                        "step follows turn 1's last, and the dotted segment between "
                        "them is the context carried into the new turn. "
                        if turn_n > 1 else
                        "The x axis will keep counting into turn 2 rather than "
                        "restarting, so the whole session reads as one curve. "
                    )
                    st.caption(
                        f"{len(points)} agent steps across {turn_n} turn(s). "
                        + across +
                        "Hover any point for the turn it belongs to, its step "
                        "within that turn, the exact token delta, and the tool that "
                        "call requested -- the rise to the next point is that "
                        "tool's result landing in the prompt. The y-intercept is "
                        "fixed overhead paid on every call (system prompt plus tool "
                        "schemas); the slope is context the agent accumulates as it "
                        "works."
                    )
                if ctx_live:
                    stamp = datetime.now().strftime("%H:%M:%S")
                    if poll_err:
                        st.caption(f"Live follow at {stamp} - MLflow poll: {poll_err}")
                    else:
                        st.caption(
                            f"Live follow on - polled at {stamp}, every "
                            f"{int(ctx_every)}s. New turns extend the curve as they "
                            "finish."
                        )

            _render_context_growth()

            section("Per-turn breakdown")
            rows = []
            # Wall-clock order and the same 1..N numbering the curve above uses, so
            # table row N is the curve's turn N. Not the agent's own
            # hermes.turn.number: it restarts after a `--resume`, which printed this
            # column as 1,1,2,3,4,5,6,1,2 for a single session.
            for i, t in enumerate(_ordered_turn_metrics(per_turn)):
                rows.append({
                    "turn": i + 1,
                    "wall_s": round(t["wall_s"], 2) if t["wall_s"] else None,
                    "steps": t["agent_steps"],
                    "time_to_first_tool_s": t["time_to_first_tool_s"],
                    "first_tool": t["first_tool_name"],
                    "ctx_first": t["context_first"],
                    "ctx_last": t["context_last"],
                    "tool_calls": t["tool_calls"],
                    "tool_failures": t["tool_failures"],
                })
            tdf = pd.DataFrame(rows)
            tdf.index = range(1, len(tdf) + 1)
            show_left_table(tdf)
    

            section("Tool outcomes")
            if metrics.get("hidden_failure_turns", 0):
                st.warning(
                    f"**The failure rate below is a lower bound.** "
                    f"{metrics['hidden_failure_turns']} of {n_turns} turn(s) report "
                    "a failed tool call in `hermes.turn.tool_outcomes` that no tool "
                    "span and no `tool_execution.csv` row captured. The plugin keys "
                    "both on `f\"{tool_name}:{task_id}\"`, so a failed call retried "
                    "inside the same step overwrites itself and only one attempt "
                    "survives. The retry is visible in the agent's console output as "
                    "a repeated `preparing tool_call…` line."
                )
            oc = metrics.get("outcome_counts", {})
            if not oc:
                st.info("No tool calls in this session.")
            else:
                odf = pd.DataFrame(
                    [{"outcome": k, "calls": v,
                      "counts_as": ("failure" if k in FAILURE_OUTCOMES
                                    else "neutral" if k in NEUTRAL_OUTCOMES
                                    else "success")}
                     for k, v in sorted(oc.items(), key=lambda kv: -kv[1])])
                odf.index = range(1, len(odf) + 1)
                c_left, c_right = st.columns([1, 1])
                with c_left:
                    show_left_table(odf)
                with c_right:
                    st.metric(
                        "Wall time in failed calls",
                        f"{metrics.get('tool_time_failed_s', 0.0):.2f}s",
                        help="Time spent on tool calls that failed. Separates a "
                             "reliability problem from a latency problem: many "
                             "cheap failures cost little wall time but still burn "
                             "agent steps, each of which costs a full LLM round trip.",
                    )

            if metrics.get("failed_calls", []):
                section("Failed calls")
                fdf = pd.DataFrame([
                    {"turn": f["turn"], "tool": f["tool"], "at_s": f["offset_s"],
                     "duration_s": f["duration_s"], "outcome": f["outcome"],
                     "error": f["error"]}
                    for f in metrics["failed_calls"]])
                fdf.index = range(1, len(fdf) + 1)
                show_left_table(fdf)
    

            with st.expander("What these four metrics mean", expanded=False):
                st.markdown(
                    "- **Agent steps** -- LLM round trips before a terminal answer. "
                    "In a serial agent this is the dominant latency term, because "
                    "each step costs a full request plus its generated tokens.\n"
                    "- **Time to first tool** -- how long the agent thinks before "
                    "acting. High is not automatically bad; it also describes a "
                    "turn where the model produced the answer itself.\n"
                    "- **Context growth per step** -- slope of input tokens across "
                    "steps. Cheap in latency when prefix caching is working, but it "
                    "sets KV-cache pressure and token cost.\n"
                    "- **Tool failure rate** -- failed calls over total, pooled "
                    "across turns. Read alongside *wall time in failed calls*: a "
                    "20% failure rate costing 0.2s is a correctness annoyance, "
                    "while one costing 40s is a latency bug."
                )


    with tab_traces:
        st.subheader("MLflow traces for this session")
        st.caption("Every prompt/turn in this session, as recorded in MLflow tracing.")

        if st.button("Load traces", key="load_traces"):
            with st.spinner("Fetching traces …"):
                tr_df, tr_err = fetch_session_traces(
                    tracking_uri, sess_id, info.get("experiment_id")
                )
            st.session_state["traces_result"] = {"df": tr_df, "err": tr_err}

        tr = st.session_state.get("traces_result")
        if tr is None:
            st.info("Click **Load traces** to fetch this session's traces from MLflow.")
        elif tr["err"]:
            st.warning(tr["err"])
        elif tr["df"].empty:
            st.info("No traces found for this session.")
        else:
            st.write(f"Found **{len(tr['df'])}** trace(s).")
            tdf = tr["df"]
            # Surface the raw column names search_traces returned, to help map
            # latency/token fields if any display empty.
            raw_cols = tdf.attrs.get("raw_columns")
            # search_traces returns newest-first (n..1); flip to oldest-first (1..n)
            # and give a 1-based row number instead of the 0-based index.
            tdf = tdf.iloc[::-1].reset_index(drop=True)
            tdf.index = range(1, len(tdf) + 1)
            if raw_cols:
                with st.expander("Debug: raw trace columns from MLflow", expanded=False):
                    st.write(raw_cols)
            if "open_in_mlflow" in tdf.columns:
                # Render numeric columns as strings to left-align them (the grid
                # right-aligns real numbers). Keep open_in_mlflow as a URL string so
                # LinkColumn stays clickable.
                disp = tdf.copy()
                for col in disp.columns:
                    if col != "open_in_mlflow" and pd.api.types.is_numeric_dtype(disp[col]):
                        disp[col] = disp[col].map(lambda v: "" if pd.isna(v) else f"{v:g}")
                st.dataframe(
                    disp,
                    width="stretch",
                    column_config={
                        "open_in_mlflow": st.column_config.LinkColumn(
                            "Open in MLflow", display_text="↗ View trace"
                        ),
                    },
                )
            else:
                # Old cached result without the link column - tell the user to reload.
                st.info("No link column found (stale cache). Click **Clear cache** in "
                        "the ⋮ menu, then **Load traces** again.")
                st.dataframe(tdf, width="stretch")


    with tab_analysis:
        st.subheader("Hermes Analysis")
        st.caption(
            "Runs the local `hermes` CLI with a prompt that analyzes this session's "
            "tool usage (tool_execution.csv) and suggests improvements."
        )

        include_traces = st.toggle(
            "Include full session traces (JSON) for more accurate, per-query analysis",
            value=True,
            help="Downloads this session's complete MLflow traces (each user query "
                 "with its full span tree - LLM calls and tool inputs/outputs) and "
                 "sends them alongside tool_execution.csv, so hermes can attribute "
                 "tool calls to the query that triggered them and compare multiple "
                 "queries. Same data as download_session_traces.py.",
        )

        # Full traces (with spans) so the analysis sees exactly what each query did.
        # These are pre-fetched at Load (loaded_full_traces); only fall back to
        # downloading here if that pre-fetch came back empty.
        full_traces = None
        if include_traces and loaded_full_traces:
            full_traces = loaded_full_traces
            st.caption(f"Including **{len(full_traces)}** full trace(s) in the analysis.")
        elif include_traces:
            tr_cached = st.session_state.get("traces_result")
            if tr_cached and not tr_cached.get("err") and not tr_cached["df"].empty:
                summary_df = tr_cached["df"]
                summary_err = ""
            else:
                with st.spinner("Resolving this session's traces …"):
                    summary_df, summary_err = fetch_session_traces(
                        tracking_uri, sess_id, info.get("experiment_id")
                    )
                if not summary_err and not summary_df.empty:
                    st.session_state["traces_result"] = {"df": summary_df, "err": ""}

            if summary_err:
                st.warning(f"Traces unavailable, analyzing CSV only: {summary_err}")
            elif summary_df.empty or "trace_id" not in summary_df.columns:
                st.warning("No trace ids for this session; analyzing CSV only.")
            else:
                trace_ids = tuple(str(t) for t in summary_df["trace_id"].dropna().tolist())
                with st.spinner(f"Downloading {len(trace_ids)} full trace(s) with spans …"):
                    full_traces, full_err = fetch_full_traces(tracking_uri, trace_ids)
                if full_err:
                    st.warning(f"Full-trace download failed, analyzing CSV only: {full_err}")
                    full_traces = None
                else:
                    st.caption(f"Including **{len(full_traces)}** full trace(s) in the analysis.")

        with st.expander("Raw tool_execution.csv (sent to hermes)", expanded=False):
            show_left_table(tool_df)
        if full_traces:
            with st.expander("Full traces JSON (sent to hermes)", expanded=False):
                st.json(full_traces)

        run_state = st.session_state.get("analysis_run")  # dict while running
        running = run_state is not None and run_state.get("proc") is not None

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Analyze with hermes", type="primary", disabled=running):
                proc, out_or_err = start_hermes_analysis(local_dir, sess_id, full_traces)
                if proc is None:
                    st.session_state["analysis_result"] = {"ok": False, "text": out_or_err}
                    st.session_state.pop("analysis_run", None)
                else:
                    st.session_state["analysis_run"] = {"proc": proc, "out_path": out_or_err}
                    st.session_state.pop("analysis_result", None)
                    st.rerun()
        with c2:
            if st.button("⏹ Stop", type="primary", disabled=not running):
                rs = st.session_state.get("analysis_run")
                if rs and rs.get("proc"):
                    try:
                        rs["proc"].terminate()
                        rs["proc"].wait(timeout=5)
                    except Exception:
                        try:
                            rs["proc"].kill()
                        except Exception:
                            pass
                    partial = read_analysis_output(rs.get("out_path", ""))
                    st.session_state["analysis_result"] = {
                        "ok": False,
                        "text": "Analysis stopped by user."
                                + (f"\n\nPartial output:\n\n{partial}" if partial else ""),
                    }
                st.session_state.pop("analysis_run", None)
                st.rerun()

        @st.fragment(run_every=2)
        def _poll_analysis():
            """Poll the running hermes subprocess on its own timer, isolated from
        the rest of the page.

        A plain time.sleep()+st.rerun() loop here reruns the WHOLE script
        every 2s while hermes works - and st.tabs() does not persist the
        selected tab across a full-page rerun, so that bounced this page back
        to the first tab (Overview, with its CPU/GPU graph) every 2 seconds
        instead of staying on the Analysis tab until the result was ready.
        st.fragment reruns only this function's body on its own schedule,
        leaving tab selection (and the rest of the page) untouched.
        """
            rs = st.session_state.get("analysis_run")
            if rs is None:
                return  # nothing running; cheap no-op until Analyze is clicked
            proc = rs["proc"]
            if proc.poll() is None:
                st.info("Running hermes analysis … click **⏹ Stop** to cancel.")
                return
            # Finished - capture output, clear running state, and do ONE
            # page-level rerun so the result renders below, outside this fragment.
            text = read_analysis_output(rs.get("out_path", ""))
            ok = proc.returncode == 0 or bool(text)
            st.session_state["analysis_result"] = {
                "ok": ok,
                "text": text or f"hermes exited with code {proc.returncode} and no output.",
            }
            st.session_state.pop("analysis_run", None)
            st.rerun()

        _poll_analysis()

        result = st.session_state.get("analysis_result")
        if result:
            if result["ok"]:
                st.markdown(result["text"])
            else:
                st.error(result["text"])
