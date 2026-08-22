#!/usr/bin/env python3
"""
Rebuild tts_executed.ipynb = the current tts.ipynb, with executed outputs.

Alignment strategy (no fabrication):
  * STRUCTURE comes wholesale from tts.ipynb, so prose, images, captions and
    ordering always match the canonical notebook.
  * OUTPUTS are REAL. They are matched to code cells by exact source text:
      - Cells whose source is unchanged keep the genuine outputs captured on the
        MI300X workshop machine (carried over from the previous executed
        notebook).
      - The Matplotlib chart cell is re-executed locally right here, because it
        is pure presentation and needs no GPU, so its output is freshly real.
  * Any code cell with no genuine matching output is left UNEXECUTED rather than
    given an invented one. The script reports exactly which, so the gap is
    visible instead of hidden.

Run:  python scripts/build_exec_notebook.py
"""
import base64
import copy
import io
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
NB = os.path.join(ROOT, "tts.ipynb")
EXEC_NB = os.path.join(ROOT, "tts_executed.ipynb")
# Genuine captured outputs from the workshop machine.
CAPTURED = os.path.join(HERE, "captured_outputs.json")


def src_of(cell):
    return "".join(cell["source"])


def executable_text(s):
    """Normalize a cell to the text that actually RUNS.

    Comments and blank lines change nothing about what a cell does, so a
    comment-only edit must not silently discard a genuine captured output and
    leave the cell blank. Everything executable is compared verbatim, so a real
    change to a command still correctly invalidates the capture.
    """
    out = []
    for line in s.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(line.rstrip())
    return "\n".join(out)


def run_chart_cell(source):
    """Execute the pure-matplotlib chart cell locally and capture real output."""
    workdir = tempfile.mkdtemp()
    runner = os.path.join(workdir, "_run.py")
    png_out = os.path.join(workdir, "chart.png")
    with open(runner, "w") as f:
        f.write(
            "import matplotlib\nmatplotlib.use('Agg')\n"
            + source
            + f"\nplt.savefig({png_out!r}, dpi=150, bbox_inches='tight',"
              " facecolor='white')\n"
        )
    proc = subprocess.run([sys.executable, runner], cwd=workdir,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"chart cell failed to execute:\n{proc.stderr}")
    with open(png_out, "rb") as f:
        png_b64 = base64.b64encode(f.read()).decode("ascii")
    return proc.stdout, png_b64


def main():
    with open(NB) as f:
        nb = json.load(f)
    with open(CAPTURED) as f:
        captured = json.load(f)   # {source_text: [output, ...]}

    out_nb = copy.deepcopy(nb)
    exec_count = 0
    report = []

    # Secondary index: executable text -> outputs. Used only when the verbatim
    # source does not match, so a comment reflow never costs a real output.
    by_exec = {}
    for k, v in captured.items():
        by_exec.setdefault(executable_text(k), v)

    for cell in out_nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = src_of(cell)

        if "matplotlib" in s and "plt.bar" in s.replace("ax.bar", "plt.bar"):
            stdout, png_b64 = run_chart_cell(s)
            exec_count += 1
            cell["execution_count"] = exec_count
            cell["outputs"] = [
                {
                    "output_type": "display_data",
                    "data": {
                        "image/png": png_b64,
                        "text/plain": ["<Figure size 1140x720 with 1 Axes>"],
                    },
                    "metadata": {},
                },
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [stdout],
                },
            ]
            report.append(("EXECUTED LOCALLY", s[:55]))
            continue

        if s in captured:
            exec_count += 1
            cell["execution_count"] = exec_count
            cell["outputs"] = captured[s]
            report.append(("REAL CAPTURE   ", s[:55]))
        elif executable_text(s) in by_exec:
            exec_count += 1
            cell["execution_count"] = exec_count
            cell["outputs"] = by_exec[executable_text(s)]
            report.append(("REAL (comments)", s[:55]))
        else:
            cell["execution_count"] = None
            cell["outputs"] = []
            report.append(("NO OUTPUT      ", s[:55]))

    with open(EXEC_NB, "w") as f:
        json.dump(out_nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"wrote {EXEC_NB}")
    print(f"cells: {len(out_nb['cells'])}")
    for status, s in report:
        print(f"  {status} | {s.strip()!r}")


if __name__ == "__main__":
    main()
