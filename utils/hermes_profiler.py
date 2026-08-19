"""Standalone Hermes Telemetry Dashboard.

A self-contained Streamlit app for exploring a Hermes agent session recorded in
MLflow by the hermes-otel plugin. Point it at an MLflow tracking server (IP +
port), pick a session id, and it will:

  1. connect to the tracking server and list every session id it can find
     (session_id is stored as an MLflow run param by the plugin),
  2. resolve the chosen session id -> run_id,
  3. download that run's ``profiling/`` artifacts (CPU/GPU timelines and the
     per-tool breakdown CSVs) locally,
  4. fetch the session's MLflow traces (one per user turn, full span trees).

The UI is organized into four tabs:

  * Overview          - CPU% + GPU% timeline with tool-execution spans; a toggle
                        swaps to a full-session span waterfall correlated with
                        CPU/GPU on one shared wall-clock axis.
  * CPU / GPU separate - the two utilization signals on their own charts, with an
                        optional side-by-side raw-CSV panel.
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
import time
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    # only when the figure actually HAS one: passing a title dict with no `text`
    # makes Plotly render the literal string "undefined" on untitled figures
    # (this bit the span waterfall).
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

# Downloaded artifacts live under a stable cache dir in $HOME (not the system
# temp dir) so they sit in one predictable place that a shutdown/cleanup step can
# purge.
PROFILING_CACHE_DIR = os.path.join(os.path.expanduser("~"), "profiling_cache")

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


@st.cache_data(show_spinner=False)
def resolve_run(tracking_uri: str, session_id: str):
    """Find the run whose params.session_id matches, across all experiments.

    Returns dict(run_id, experiment, experiment_id, artifact_uri), or None if no
    run has that session_id.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments()
    if not experiments:
        return None
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in experiments],
        filter_string=f"params.session_id = '{session_id}'",
        max_results=1,
        order_by=["start_time DESC"],
    )
    if not runs:
        return None
    run = runs[0]
    exp_name = next(
        (e.name for e in experiments if e.experiment_id == run.info.experiment_id),
        run.info.experiment_id,
    )
    return {
        "run_id": run.info.run_id,
        "experiment": exp_name,
        "experiment_id": run.info.experiment_id,
        "artifact_uri": run.info.artifact_uri,
    }


@st.cache_data(show_spinner=False)
def fetch_session_ids(tracking_uri: str):
    """Return all distinct session ids on the tracking server, newest first.

    session_id is stored as a run param (see the plugin's mlflow_hooks.py), so we
    scan runs across every experiment, keep the most recent start_time per session
    id, and return them ordered newest-first for the dropdown.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments()
    if not experiments:
        return []
    latest = {}  # session_id -> newest start_time seen
    token = None
    while True:
        runs = client.search_runs(
            experiment_ids=[e.experiment_id for e in experiments],
            filter_string="attributes.status != 'DELETED'",
            max_results=1000,
            order_by=["start_time DESC"],
            page_token=token,
        )
        for run in runs:
            sid = run.data.params.get("session_id")
            if not sid:
                continue
            start = run.info.start_time or 0
            if sid not in latest or start > latest[sid]:
                latest[sid] = start
        token = getattr(runs, "token", None)
        if not token:
            break
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
    """Prompt that points hermes at the on-disk data files to analyze.

    The prompt is passed to `hermes -z <PROMPT>` as a single CLI argument, which
    the OS caps at ~128 KB (Linux MAX_ARG_STRLEN) - inlining a dozen full traces
    would blow past it ("Argument list too long"). So we always write the data to
    disk and pass only these short path references. Under --yolo one-shot mode
    hermes still loads its file-reading tools, so it can open the paths itself.
    """
    prompt = (
        "You are a performance analyst. The telemetry for a Hermes agent session "
        "is on disk. Read these files with your file tools and analyze them "
        "directly. Base your analysis ONLY on their contents.\n\n"
        f"1. tool_breakdown.csv (one row per tool call): {tool_csv_path}\n"
        "   Columns: turn, tool_name, input, output, timestamp, "
        "start_time_unix_nano, elapsed_s, duration_s, cpu_avg_pct, cpu_peak_pct, "
        "gpu_avg_pct, gpu_peak_pct. duration_s is the tool's execution time; "
        "cpu_avg_pct/gpu_avg_pct are the average CPU/GPU during it (CPU is "
        "hermes+children, excludes the vLLM model server).\n"
    )
    if traces_path:
        prompt += (
            f"2. traces.json (JSON array of full MLflow traces, one per user "
            f"query, spans included): {traces_path}\n\n"
            "JOIN KEY: each trace has a span attribute `hermes.turn.number` whose "
            "value equals the `turn` column in tool_breakdown.csv. The trace with "
            "hermes.turn.number = N owns every CSV row where turn = N (one query = "
            "one turn = one trace). Do NOT guess by timestamp; use the turn "
            "number. Analyze each query SEPARATELY, then compare them.\n\n"
        )
    else:
        prompt += "\n"
    prompt += (
        "Please report:\n"
        "1. The hotspot tools/turns (which tools dominate time and resource use) and why.\n"
        "2. Redundant, repeated, or inefficient tool usage patterns you notice.\n"
        "3. Concrete, actionable improvements to the tool usage (e.g. batching, "
        "avoiding repeated calls, cheaper alternatives).\n"
    )
    if traces_path:
        prompt += (
            "4. A per-query breakdown: for each user query (trace), its dominant "
            "tools, total tool time, and one specific improvement.\n"
        )
    prompt += "Keep it concise and specific to this data."
    return prompt


def start_hermes_analysis(local_dir: str, session_id: str, full_traces=None):
    """Start hermes analysis as a non-blocking subprocess.

    Returns (proc, out_path) or (None, error_message). Output is streamed to a
    temp file so it can be read after the process finishes, and so a blocking
    subprocess.run() never freezes the Streamlit UI and leaves the Stop button
    unclickable. The -z flag gives quiet output.

    The data is always written to disk and referenced by path (never inlined), so
    the prompt stays tiny regardless of how many traces there are and can never
    hit the OS command-line arg limit. tool_breakdown.csv already sits in
    local_dir (under PROFILING_CACHE_DIR, see download_profiling), so it is
    referenced in place; the traces are reused from the shared per-session cache
    file (traces_cache_path) that Load already wrote, so both the dashboard and
    this analysis read the same traces.json. Everything lives under
    PROFILING_CACHE_DIR so it is purged together on cleanup.
    """
    tool_csv_path = os.path.join(local_dir, "tool_breakdown.csv")

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
    ns), the same reference the poller writes into cpu/gpu_timeline.csv (ts_abs).
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
    # tool_breakdown.csv's start_time_unix_nano column - the same epoch-ns
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
# UI
# ---------------------------------------------------------------------------

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
    with st.spinner("Resolving session → run …"):
        try:
            info = resolve_run(tracking_uri, session_id.strip())
        except Exception as e:
            st.error(f"Could not reach MLflow at {tracking_uri}: {e}")
            st.stop()
    if not info:
        st.error(f"No run found with session_id = '{session_id}'.")
        st.stop()

    with st.spinner("Downloading profiling artifacts …"):
        try:
            local_dir = download_profiling(tracking_uri, info["run_id"], session_id.strip())
        except Exception as e:
            st.error(f"Could not download '{ARTIFACT_DIR}' artifacts: {e}")
            st.stop()

    # Pre-fetch this session's full traces (JSON with all spans) up front so both
    # the Tab-1 timeline waterfall and the Analysis tab have them ready without an
    # extra click. Traces are optional telemetry, so any failure is non-fatal.
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

    st.session_state["loaded"] = {
        "tracking_uri": tracking_uri,
        "info": info,
        "session_id": session_id.strip(),
        "local_dir": local_dir,
        "full_traces": full_traces,
        "turn_count": turn_count,
        "total_latency_ms": total_latency_ms,
        "cpu_df": parse_timestamps(read_csv(os.path.join(local_dir, "cpu_timeline.csv"))),
        "gpu_df": parse_timestamps(read_csv(os.path.join(local_dir, "gpu_timeline.csv"))),
        "tool_df": parse_timestamps(read_csv(os.path.join(local_dir, "tool_breakdown.csv"))),
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
if not _data or "local_dir" not in _data or "session_id" not in _data:
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

st.write(f"**Tracking URI:** `{tracking_uri}`")
st.success(f"Found run `{info['run_id']}` in experiment `{info['experiment']}`.")

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

art_uri = info["artifact_uri"] or ""
if not art_uri.startswith("mlflow-artifacts:"):
    st.warning(
        f"Artifact URI is `{art_uri}` - this looks like a direct/local path, not a "
        "server-proxied `mlflow-artifacts:/…` URI. Remote download may fail. If so, "
        "start the server with `--serve-artifacts --artifacts-destination <path>`."
    )

if cpu_df.empty and gpu_df.empty:
    st.warning("No CPU/GPU timeline data found in the downloaded artifacts.")
    st.stop()

tab_overview, tab_separate, tab_traces, tab_analysis = st.tabs(
    ["Overview", "CPU / GPU separate", "Traces", "Analysis"]
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
            st.write("No tool_breakdown.csv data.")
        else:
            # Show the row number starting at 1 instead of the 0-based index.
            _disp = tool_df.copy()
            _disp.index = range(1, len(_disp) + 1)
            show_left_table(_disp)

with tab_separate:
    sources = {
        "CPU Usage": ("cpu_timeline.csv", cpu_df),
        "GPU usage": ("gpu_timeline.csv", gpu_df),
        "Tool Track": ("tool_breakdown.csv", tool_df),
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
        "tool usage (tool_breakdown.csv) and suggests improvements."
    )

    include_traces = st.toggle(
        "Include full session traces (JSON) for more accurate, per-query analysis",
        value=True,
        help="Downloads this session's complete MLflow traces (each user query "
             "with its full span tree - LLM calls and tool inputs/outputs) and "
             "sends them alongside tool_breakdown.csv, so hermes can attribute "
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

    with st.expander("Raw tool_breakdown.csv (sent to hermes)", expanded=False):
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

    # While running: poll the process; refresh ~every 2s so the Stop button stays
    # responsive (subprocess.Popen is non-blocking, unlike subprocess.run).
    if running:
        rs = st.session_state["analysis_run"]
        proc = rs["proc"]
        if proc.poll() is None:
            st.info("Running hermes analysis … click **⏹ Stop** to cancel.")
            time.sleep(2)
            st.rerun()
        else:
            # Finished - capture output and clear running state.
            text = read_analysis_output(rs.get("out_path", ""))
            ok = proc.returncode == 0 or bool(text)
            st.session_state["analysis_result"] = {
                "ok": ok,
                "text": text or f"hermes exited with code {proc.returncode} and no output.",
            }
            st.session_state.pop("analysis_run", None)
            st.rerun()

    result = st.session_state.get("analysis_result")
    if result:
        if result["ok"]:
            st.markdown(result["text"])
        else:
            st.error(result["text"])
