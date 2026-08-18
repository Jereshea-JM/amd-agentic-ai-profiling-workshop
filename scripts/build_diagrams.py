#!/usr/bin/env python3
"""
Build AMD-branded, light-theme concept diagrams for the TTS profiling workshop.

Each diagram is authored as SVG (editable source) and rasterized to PNG with
cairosvg at scale=2 for crisp high-DPI output. Both .svg and .png are written to
notebooks assets so the diagrams are reproducible and diffable.

Palette and fonts are grounded in the actual AMD product:
  - AMD Red (official brand accent)        #ED1C24
  - Ink / near-black text                   #1A1A1A
  - Dashboard CPU blue (sampled)            #2E6DB4
  - Dashboard TTS-highlight orange (sampled)#F08418
  - Kokoro/local teal                       #1EAAB4
  - Panel fill / hairlines                  #F4F5F7 / #D9DCE1
Font stack is Arial-metric-compatible (Liberation Sans) so it reads as the AMD
web font family on any machine.
"""
import os
import cairosvg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "assets", "diagrams"))
os.makedirs(OUT, exist_ok=True)

# ---- AMD brand tokens -------------------------------------------------------
AMD_RED   = "#ED1C24"
INK       = "#1A1A1A"
SUBINK    = "#5B6270"
BLUE      = "#2E6DB4"   # CPU (from dashboard)
ORANGE    = "#F08418"   # TTS-highlight span (from dashboard)
TEAL      = "#1EAAB4"   # local / Kokoro
GOLD      = "#C1A968"
PANEL     = "#F4F5F7"
LINE      = "#D9DCE1"
WHITE     = "#FFFFFF"
GREEN     = "#2E8B57"
FONT = "Liberation Sans, Arial, Helvetica, sans-serif"
MONO = "DejaVu Sans Mono, Consolas, monospace"

DEFS = f"""
  <defs>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0B1020" flood-opacity="0.12"/>
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8"
            markerHeight="8" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/>
    </marker>
    <marker id="arrowred" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8"
            markerHeight="8" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{AMD_RED}"/>
    </marker>
  </defs>
"""


def svg(w, h, body, bg=WHITE):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">'
        f'{DEFS}<rect x="0" y="0" width="{w}" height="{h}" rx="14" fill="{bg}"/>'
        f'{body}</svg>'
    )


def card(x, y, w, h, fill, stroke, rx=12, shadow=True):
    sh = ' filter="url(#shadow)"' if shadow else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{sh}/>')


def text(x, y, s, size=17, fill=INK, weight="normal", anchor="middle",
         family=FONT, spacing=None):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'font-family="{family}"{ls}>{s}</text>')


def line(x1, y1, x2, y2, color=INK, w=2.2, arrow=True, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    mk = ' marker-end="url(#arrow)"' if arrow else ""
    if arrow and color == AMD_RED:
        mk = ' marker-end="url(#arrowred)"'
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{w}" stroke-linecap="round"{d}{mk}/>')


def badge(cx, cy, label, color):
    """small pill tag"""
    w = 12 + len(label) * 7.2
    return (f'<rect x="{cx-w/2}" y="{cy-11}" width="{w}" height="22" rx="11" '
            f'fill="{color}" opacity="0.14"/>' +
            text(cx, cy + 4, label, size=12.5, fill=color, weight="bold"))


def save(name, markup):
    svg_path = os.path.join(OUT, name + ".svg")
    png_path = os.path.join(OUT, name + ".png")
    with open(svg_path, "w") as f:
        f.write(markup)
    cairosvg.svg2png(bytestring=markup.encode(), write_to=png_path, scale=2)
    kb = os.path.getsize(png_path) / 1024
    print(f"  {name}.png  {kb:6.1f} KB")


# =============================================================================
# 1. PIPELINE  (Text File -> Hermes Agent -> TTS Tool -> Audio Output)
# =============================================================================
def d_pipeline():
    W, H = 900, 250
    b = [text(W/2, 40, "The workflow we profile", size=22, weight="bold", fill=INK)]
    b.append(text(W/2, 64, "Text in, spoken audio out, with every step measured",
                  size=14, fill=SUBINK))
    stages = [
        ("input_text.txt", "Input passage", PANEL, INK, "\ud83d\udcc4"),
        ("Hermes Agent", "Plans &amp; picks tools", "#FDECEC", AMD_RED, "\u2699"),
        ("TTS Tool", "Edge \u2192 Kokoro", "#EAF2FB", BLUE, "\ud83d\udd0a"),
        ("Audio output", "output_audio", "#E9F7F8", TEAL, "\u266a"),
    ]
    n = len(stages)
    cw, ch = 178, 92
    gap = (W - 40 - n * cw) / (n - 1)
    y = 108
    x = 20
    xs = []
    for i, (title, sub, fill, accent, icon) in enumerate(stages):
        xs.append((x, x + cw))
        b.append(card(x, y, cw, ch, fill, accent))
        b.append(f'<rect x="{x}" y="{y}" width="6" height="{ch}" rx="3" fill="{accent}"/>')
        b.append(text(x + cw/2, y + 38, title, size=16.5, weight="bold", fill=INK))
        b.append(text(x + cw/2, y + 62, sub, size=13, fill=SUBINK))
        if i < n - 1:
            ax1 = x + cw + 6
            ax2 = x + cw + gap - 6
            b.append(line(ax1, y + ch/2, ax2, y + ch/2, color=INK, w=2.4))
        x += cw + gap
    b.append(text(W/2, 232,
                  "Focus of this workshop: profiling &amp; optimization. TTS is the example.",
                  size=13.5, fill=AMD_RED, weight="bold"))
    save("01_pipeline", svg(W, H, "".join(b)))


# =============================================================================
# 2. ARCHITECTURE  (what helper.sh brings up)
# =============================================================================
def d_architecture():
    W, H = 900, 470
    b = [text(W/2, 40, "What one command starts for you", size=22, weight="bold")]
    b.append(text(W/2, 64,
                  "bash helper.sh: the full observability backend, no manual setup",
                  size=14, fill=SUBINK, family=FONT))

    # Compute / agent brain
    b.append(card(40, 96, 380, 120, "#FDECEC", AMD_RED))
    b.append(text(230, 124, "Agent runtime", size=16, weight="bold", fill=AMD_RED))
    b.append(text(230, 150, "Hermes Agent", size=15, weight="bold", fill=INK))
    b.append(text(230, 172, "vLLM \u00b7 Muse-Glimmer-30B", size=13, fill=SUBINK))
    b.append(text(230, 196, "plans, reasons, picks and calls tools", size=12.5, fill=SUBINK))

    # Telemetry pipe
    b.append(card(480, 96, 380, 120, "#EAF2FB", BLUE))
    b.append(text(670, 124, "Telemetry", size=16, weight="bold", fill=BLUE))
    b.append(text(670, 150, "hermes-otel (patched)", size=15, weight="bold", fill=INK))
    b.append(text(670, 172, "OTel spans + psutil CPU%", size=13, fill=SUBINK))
    b.append(text(670, 196, "+ AMD GPU metrics, every 0.1s", size=12.5, fill=SUBINK))
    b.append(text(450, 146, "instruments", size=10, fill=SUBINK, anchor="middle"))
    b.append(line(420, 156, 480, 156, color=INK))

    # Kokoro server (right of agent, feeds work)
    b.append(card(40, 240, 380, 96, "#E9F7F8", TEAL))
    b.append(text(230, 268, "Kokoro TTS server", size=15, weight="bold", fill=TEAL))
    b.append(text(230, 292, "FastAPI + Uvicorn on MI300X", size=13, fill=SUBINK))
    b.append(text(230, 314, "model stays resident in GPU memory", size=12.5, fill=SUBINK))
    # agent calls Kokoro as a tool
    b.append(line(230, 216, 230, 240, color=INK))
    b.append(text(292, 232, "calls as a tool", size=11, fill=SUBINK, anchor="start"))

    # MLflow
    b.append(card(480, 240, 380, 96, PANEL, INK))
    b.append(text(670, 268, "MLflow tracking server", size=15, weight="bold", fill=INK))
    b.append(text(670, 292, "records every run :5004", size=13, fill=SUBINK))
    b.append(text(670, 314, "timings + hardware metrics", size=12.5, fill=SUBINK))
    b.append(text(730, 232, "records to", size=11, fill=SUBINK, anchor="start"))
    b.append(line(670, 216, 670, 240, color=INK))

    # Dashboard (bottom, spanning)
    b.append(card(220, 372, 460, 78, "#FDECEC", AMD_RED))
    b.append(text(450, 400, "Telemetry dashboard  \u00b7  Streamlit :8501",
                  size=16, weight="bold", fill=AMD_RED))
    b.append(text(450, 426, "one clear view of the run: spans, CPU/GPU timeline, tool breakdown",
                  size=13, fill=SUBINK))
    # MLflow -> dashboard (the dashboard reads MLflow)
    b.append(line(600, 336, 480, 372, color=INK))
    b.append(line(230, 336, 400, 372, color=INK, dash="5 4"))
    save("02_architecture", svg(W, H, "".join(b)))


# =============================================================================
# 3. PROFILING LOOP  (the always-the-same loop)
# =============================================================================
def d_loop():
    W, H = 900, 260
    b = [text(W/2, 40, "The observability-driven loop", size=22, weight="bold")]
    b.append(text(W/2, 64, "The same five moves for any agent task, not just TTS",
                  size=14, fill=SUBINK))
    steps = [
        ("1", "Run", "the agent", AMD_RED),
        ("2", "Fetch", "in dashboard", BLUE),
        ("3", "Inspect", "spans &amp; GPU", ORANGE),
        ("4", "Optimize", "the slow tool", TEAL),
        ("5", "Measure", "again", GREEN),
    ]
    n = len(steps)
    cw, ch = 150, 96
    gap = (W - 40 - n * cw) / (n - 1)
    y = 108
    x = 20
    for i, (num, title, sub, accent) in enumerate(steps):
        b.append(card(x, y, cw, ch, WHITE, accent))
        b.append(f'<circle cx="{x+26}" cy="{y+26}" r="15" fill="{accent}"/>')
        b.append(text(x + 26, y + 31, num, size=16, weight="bold", fill=WHITE))
        b.append(text(x + cw/2 + 10, y + 30, title, size=16, weight="bold", fill=INK))
        b.append(text(x + cw/2, y + 62, sub, size=13, fill=SUBINK))
        if i < n - 1:
            b.append(line(x + cw + 4, y + ch/2, x + cw + gap - 4, y + ch/2,
                          color=INK, w=2.4))
        x += cw + gap
    # loop-back arc
    b.append(f'<path d="M {W-40} {y+ch+8} q 0 34 -{W-80} 0" fill="none" '
             f'stroke="{AMD_RED}" stroke-width="2.2" stroke-dasharray="6 5" '
             f'marker-end="url(#arrowred)"/>')
    b.append(text(W/2, y + ch + 46, "repeat until the bottleneck is gone",
                  size=13, fill=AMD_RED, weight="bold"))
    save("03_loop", svg(W, H, "".join(b)))


# =============================================================================
# 4. THREE APPROACHES journey
# =============================================================================
def d_journey():
    W, H = 900, 300
    b = [text(W/2, 40, "Three approaches, one input", size=22, weight="bold")]
    b.append(text(W/2, 64, "From cloud convenience to a fully local, GPU-optimized tool",
                  size=14, fill=SUBINK))
    cols = [
        ("Edge TTS", "Cloud baseline", ORANGE, [
            "Default provider, zero setup",
            "5,000-char input cap",
            "Text leaves the machine",
        ]),
        ("Kokoro \u00b7 sequential", "Local baseline", BLUE, [
            "Runs on MI300X, data stays local",
            "One sentence per GPU pass",
            "Correct, but under-uses the GPU",
        ]),
        ("Kokoro \u00b7 batched", "Local, optimized", TEAL, [
            "Many sentences per GPU pass",
            "Length bucketing + masks",
            "Same audio, far higher throughput",
        ]),
    ]
    n = len(cols)
    cw, ch = 268, 176
    gap = (W - 40 - n * cw) / (n - 1)
    y = 92
    x = 20
    for i, (title, tag, accent, bullets) in enumerate(cols):
        b.append(card(x, y, cw, ch, WHITE, accent))
        b.append(f'<rect x="{x}" y="{y}" width="{cw}" height="46" rx="12" fill="{accent}"/>')
        b.append(f'<rect x="{x}" y="{y+24}" width="{cw}" height="22" fill="{accent}"/>')
        b.append(text(x + cw/2, y + 30, title, size=17, weight="bold", fill=WHITE))
        b.append(badge(x + cw/2, y + 68, tag, accent))
        ty = y + 98
        for bl in bullets:
            b.append(f'<circle cx="{x+22}" cy="{ty-4}" r="3" fill="{accent}"/>')
            b.append(text(x + 34, ty, bl, size=12.8, fill=INK, anchor="start"))
            ty += 24
        if i < n - 1:
            b.append(line(x + cw + 4, y + ch/2, x + cw + gap - 4, y + ch/2,
                          color=AMD_RED, w=2.6))
        x += cw + gap
    save("04_journey", svg(W, H, "".join(b)))


# =============================================================================
# 5. SEQUENTIAL vs BATCHED  (the optimization, visualized)
# =============================================================================
def d_batching():
    W, H = 900, 366
    b = [text(W/2, 40, "Why batching wins on the GPU", size=22, weight="bold")]
    b.append(text(W/2, 64, "Same sentences, far fewer GPU launches", size=14, fill=SUBINK))

    def gpu(x, y, label, accent):
        b.append(card(x, y, 150, 64, "#101418", accent, shadow=True))
        b.append(text(x + 75, y + 30, "MI300X GPU", size=13.5, weight="bold", fill="#EDEFF3"))
        b.append(text(x + 75, y + 50, label, size=12, fill="#9AA3B2"))

    # Sequential row
    b.append(text(40, 108, "Sequential mode", size=16, weight="bold", fill=BLUE, anchor="start"))
    b.append(text(40, 128, "one sentence per forward pass, GPU mostly idle",
                  size=12.5, fill=SUBINK, anchor="start"))
    sx = 40
    sy = 142
    for i in range(5):
        b.append(card(sx, sy, 42, 34, "#EAF2FB", BLUE, rx=7, shadow=False))
        b.append(text(sx + 21, sy + 22, f"s{i+1}", size=12, fill=BLUE, weight="bold"))
        b.append(line(sx + 21, sy + 34, sx + 21, sy + 58, color=BLUE, w=1.8))
        b.append(card(sx, sy + 58, 42, 22, PANEL, LINE, rx=6, shadow=False))
        b.append(text(sx + 21, sy + 73, "pass", size=9.5, fill=SUBINK))
        sx += 58
    gpu(560, sy + 20, "low utilization", BLUE)
    b.append(line(330, sy + 40, 556, sy + 52, color=BLUE, dash="5 4"))
    b.append(badge(635, sy + 100, "5 GPU launches", BLUE))

    # Batched row
    by = 250
    b.append(text(40, by - 8, "Batched mode", size=16, weight="bold", fill=TEAL, anchor="start"))
    b.append(card(40, by + 6, 274, 40, "#E9F7F8", TEAL, rx=8, shadow=False))
    for i in range(5):
        bx = 52 + i * 52
        b.append(card(bx, by + 12, 42, 28, WHITE, TEAL, rx=6, shadow=False))
        b.append(text(bx + 21, by + 31, f"s{i+1}", size=12, fill=TEAL, weight="bold"))
    b.append(text(177, by + 62, "grouped + length-bucketed", size=11.5, fill=SUBINK))
    b.append(line(316, by + 26, 556, by + 40, color=TEAL, w=2.4))
    gpu(560, by + 8, "high utilization", TEAL)
    b.append(badge(635, by + 80, "1 GPU launch", TEAL))
    save("05_batching", svg(W, H, "".join(b)))


if __name__ == "__main__":
    print("Building AMD-branded diagrams ->", OUT)
    d_pipeline()
    d_architecture()
    d_loop()
    d_journey()
    d_batching()
    print("done.")
