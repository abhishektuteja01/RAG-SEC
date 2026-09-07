"""Producer for `images/results_chart.png` -- the arm-progression chart in the README.

Produces: images/results_chart.png (recall@10 by retrieval arm, dev split, with the
held-out test result marked separately).

Reads, and RE-DERIVES every plotted number at generation time -- nothing is typed in:

  Arm 1 (dev)   data/day3_arm1_dev_results.json    mean/stderr recomputed over `per_question`
  Arm 2 (dev)   data/day4_arm2_dev_results.json    mean/stderr recomputed over `per_question`
  Arm 3 rows    data/retr7_rr_{dev,test}_scores.jsonl, replayed through
                `scripts/pipeline/05_arm3_rerank.py score` -- the same command the README
                publishes, run as a subprocess so this file cannot drift from the scorer.

Arms 1-2 cannot be replayed offline: they query Postgres live, which a fresh clone does not
have (`scripts/README.md` phase 04). Their per-question recall vector is the furthest-upstream
artifact that survives without the database, so that is what is aggregated here -- not the
`metrics` block sitting next to it, which is only re-read as a self-consistency guard.

Every derived value is then cross-checked against `DECISIONS.md`'s "Current baseline
(post-RETR-7 re-index -- RETR-39)" table. A mismatch beyond half a unit in the published last
digit ABORTS. The chart plots the derived value, never the parsed one: DECISIONS.md is the
check, the artifacts are the source.

Usage (matplotlib is deliberately NOT a project dependency -- it is presentation only and
`uv run` syncs the default group into the venv that long eval runs execute from):

    uv run --with matplotlib scripts/archive/results_chart.py

Takes ~3-4 min: the two Arm 3 rows are two full replays of the stored rerank scores
(~90 s each). No GPU, no database, no API key, no spend. `--reuse-scored` skips the replays
and reads the scorer's committed sidecars instead, for a layout-only edit.

Traps:

- **The old chart in git history predates RETR-35's coverage-based labels.** Its numbers are
  not comparable to anything current. This script exists because that chart had no producer
  and so could not be corrected -- only deleted.
- **The DECISIONS.md parse is a CHECK, not a data source.** This project's recurring bug class
  (RETR-24, RETR-30, AGENT-16) is code that assumes a shape the producer of an artifact never
  promised -- so the parser asserts it found all five expected rows and dies otherwise, and a
  reformatted table breaks the build loudly rather than yielding a plausible wrong bar.
- **A `*_scores.jsonl` is not in rank order.** It is never read here directly; the subprocess
  reads it through `rag_sec.eval.load_ranking`, which sorts on load (AGENT-16).
- **Four bars are dev and one is test.** They are different question sets (n=1235 vs 1545
  scored), so the last two bars are NOT a before/after -- they are the same configuration
  measured on two splits. The chart says so in three places; keep all three if you edit it.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT_PATH = REPO / "images" / "results_chart.png"
SCORER = REPO / "scripts" / "pipeline" / "05_arm3_rerank.py"
DECISIONS = REPO / "DECISIONS.md"

# The five bars, top to bottom. `decisions_row` is the exact label in DECISIONS.md's baseline
# table that each one must agree with; `source` names how the value is derived.
ARM1_RESULTS = DATA / "day3_arm1_dev_results.json"
ARM2_RESULTS = DATA / "day4_arm2_dev_results.json"
DEV_SCORES = DATA / "retr7_rr_dev_scores.jsonl"
TEST_SCORES = DATA / "retr7_rr_test_scores.jsonl"
# Written by `05_arm3_rerank.py score --out`; only read under --reuse-scored.
DEV_SCORED = DATA / "retr7_arm3_dev_results.json"
TEST_SCORED = DATA / "retr7_arm3_test_results.json"

BASELINE_HEADING = "## Current baseline"
# Published means are 3 dp, so anything within half of the last digit is the same number.
TOLERANCE = 0.0005

# DECISIONS.md rows whose ± disagrees with the artifact it came from. Listed, not silently
# tolerated: the chart draws the ARTIFACT value, and the run prints this so the doc gets
# fixed. Anything NOT listed here aborts the run.
# Empty on purpose. The one entry this ever held -- the TEST row's stale +/- 0.012 -- was
# fixed in DECISIONS.md on 2026-09-07, so every published value now matches its artifact.
KNOWN_DOC_DISCREPANCIES: dict[tuple[str, str], str] = {}


# --- Deriving the numbers ----------------------------------------------------------------


def mean_stderr(values: list[float]) -> tuple[float, float]:
    """Same definition as `rag_sec.eval.mean_and_stderr` (sample stdev / sqrt n)."""
    if len(values) < 2:
        raise SystemExit(f"need >=2 observations to take a stderr, got {len(values)}")
    return statistics.fmean(values), statistics.stdev(values) / len(values) ** 0.5


def from_per_question(path: Path) -> tuple[float, float, int]:
    """Aggregate an Arm 1/2 results JSON from its per-question recall vector.

    The file also carries a precomputed `metrics` block. It is deliberately NOT used as the
    value -- it is compared against the recomputation, so a hand-edited summary next to
    untouched per-question data fails here instead of reaching the chart.
    """
    if not path.exists():
        raise SystemExit(f"missing artifact {path} -- cannot derive this bar; refusing to guess")
    doc = json.loads(path.read_text())
    per = doc.get("per_question")
    if not isinstance(per, list) or not per:
        raise SystemExit(f"{path}: expected a non-empty `per_question` list, got {type(per)}")
    try:
        values = [float(q["recall_10"]) for q in per]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"{path}: `per_question` entries lack a numeric recall_10 ({exc})")
    mean, stderr = mean_stderr(values)

    stored = doc.get("metrics", {}).get("recall_10")
    if not isinstance(stored, dict) or "mean" not in stored:
        raise SystemExit(f"{path}: no `metrics.recall_10.mean` to cross-check against")
    if abs(stored["mean"] - mean) > 1e-9 or abs(stored["stderr"] - stderr) > 1e-9:
        raise SystemExit(
            f"{path}: the file's own summary disagrees with its per-question data "
            f"(summary {stored['mean']:.6f} ± {stored['stderr']:.6f}, "
            f"recomputed {mean:.6f} ± {stderr:.6f}) -- one of them is stale"
        )
    if doc.get("n") not in (None, len(values)):
        raise SystemExit(f"{path}: declares n={doc['n']} but carries {len(values)} questions")
    return mean, stderr, len(values)


def replay_arm3(scores: Path, split: str, reuse: Path | None) -> dict:
    """Score a stored rerank-score file with the pipeline's own scorer, and return its table.

    Run as a subprocess rather than reimplemented: the loop that resolves gold labels, the
    cell definitions and the rank-order fix all live in `05_arm3_rerank.py`/`rag_sec.eval`,
    and a second copy of them here is exactly how this repo's provenance bugs start.
    """
    if reuse is not None:
        if not reuse.exists():
            raise SystemExit(f"--reuse-scored: {reuse} does not exist")
        doc = json.loads(reuse.read_text())
        if Path(doc.get("scores", "")).name != scores.name:
            raise SystemExit(
                f"{reuse} was produced from {doc.get('scores')!r}, not {scores} -- "
                "stale sidecar, refusing to plot it"
            )
    else:
        if not scores.exists():
            raise SystemExit(f"missing {scores} -- the Arm 3 rows cannot be replayed")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / f"{split}.json"
            cmd = [sys.executable, str(SCORER), "score", "--scores", str(scores),
                   "--split", split, "--out", str(out)]
            print(f"replaying {scores.name} ({split}) -- ~90 s ...", flush=True)
            proc = subprocess.run(cmd, cwd=REPO)
            if proc.returncode != 0:
                raise SystemExit(f"scorer failed ({' '.join(cmd)})")
            doc = json.loads(out.read_text())

    if doc.get("split") != split:
        raise SystemExit(f"scorer returned split={doc.get('split')!r}, expected {split!r}")
    for cell in ("unfiltered_raw", "filtered_stripped"):
        if cell not in doc.get("cells", {}):
            raise SystemExit(f"scorer output has no `{cell}` cell for {split}")
    return doc


# --- Cross-check against DECISIONS.md ----------------------------------------------------

_ROW_RE = re.compile(r"^\|(?P<label>[^|]+)\|(?P<recall10>[^|]+)\|")
_VALUE_RE = re.compile(r"^(?P<mean>\d\.\d+)(?:\s*±\s*(?P<stderr>\d\.\d+))?$")


def parse_baseline_table() -> dict[str, dict[str, float | None]]:
    """Pull `label -> {mean, stderr}` out of DECISIONS.md's current-baseline table.

    Deliberately strict: it takes the FIRST markdown table under the current-baseline
    heading, and the caller asserts the exact set of rows it needs. A reformat, a moved
    heading or a renamed row stops the build; it never yields a partial table.
    """
    if not DECISIONS.exists():
        raise SystemExit(f"missing {DECISIONS}")
    lines = DECISIONS.read_text().splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(BASELINE_HEADING)]
    if len(starts) != 1:
        raise SystemExit(
            f"expected exactly one '{BASELINE_HEADING}...' heading in DECISIONS.md, "
            f"found {len(starts)}"
        )

    rows: dict[str, dict[str, float | None]] = {}
    seen_table = False
    for ln in lines[starts[0] + 1:]:
        if ln.startswith("## "):
            break
        if not ln.startswith("|"):
            if seen_table:
                break  # table ended; do not pick up a later one under the same heading
            continue
        seen_table = True
        m = _ROW_RE.match(ln)
        if not m:
            raise SystemExit(f"unparseable baseline-table row in DECISIONS.md: {ln!r}")
        label = m.group("label").replace("**", "").strip()
        cell = m.group("recall10").replace("**", "").strip()
        if set(cell) <= {"-", ":", " "} or not cell:
            continue  # header separator
        v = _VALUE_RE.match(cell)
        if not v:
            if label.lower().startswith("arm"):
                continue  # column header
            raise SystemExit(f"unparseable recall@10 cell in DECISIONS.md: {cell!r} ({label!r})")
        rows[label] = {
            "mean": float(v.group("mean")),
            "stderr": float(v.group("stderr")) if v.group("stderr") else None,
        }
    if not rows:
        raise SystemExit("found the current-baseline heading but no data rows under it")
    return rows


def check(published: dict, label: str, mean: float, stderr: float, problems: list[str]) -> None:
    if label not in published:
        raise SystemExit(
            f"DECISIONS.md's baseline table has no row {label!r} "
            f"(rows found: {sorted(published)}) -- the table moved; fix this script"
        )
    row = published[label]
    if abs(row["mean"] - mean) > TOLERANCE:
        problems.append(
            f"{label}: recall@10 derived {mean:.4f}, DECISIONS.md publishes {row['mean']:.3f}"
        )
    if row["stderr"] is None:
        return
    if abs(row["stderr"] - stderr) > TOLERANCE:
        note = KNOWN_DOC_DISCREPANCIES.get((label, "stderr"))
        msg = f"{label}: stderr derived {stderr:.4f}, DECISIONS.md publishes {row['stderr']:.3f}"
        if note is None:
            problems.append(msg)
        else:
            print(f"KNOWN DOC DISCREPANCY -- {msg}  ({note})", file=sys.stderr)


def collect(reuse: bool) -> tuple[list[tuple], int, int]:
    a1_mean, a1_err, a1_n = from_per_question(ARM1_RESULTS)
    a2_mean, a2_err, a2_n = from_per_question(ARM2_RESULTS)
    dev = replay_arm3(DEV_SCORES, "dev", DEV_SCORED if reuse else None)
    test = replay_arm3(TEST_SCORES, "test", TEST_SCORED if reuse else None)

    def cell(doc, name):
        c = doc["cells"][name]["recall_10"]
        return float(c["mean"]), float(c["stderr"])

    a3_mean, a3_err = cell(dev, "unfiltered_raw")
    a3f_mean, a3f_err = cell(dev, "filtered_stripped")
    t_mean, t_err = cell(test, "filtered_stripped")
    n_dev, n_test = int(dev["n_scored"]), int(test["n_scored"])

    if not a1_n == a2_n == n_dev:
        raise SystemExit(
            f"dev bars disagree on the question count (arm1 {a1_n}, arm2 {a2_n}, "
            f"arm3 {n_dev}) -- these bars are not comparable"
        )

    bars = [
        ("Arm 1 — dense", "dev", a1_mean, a1_err, "1 — dense"),
        ("Arm 2 — + BM25/RRF", "dev", a2_mean, a2_err, "2 — +BM25/RRF"),
        ("Arm 3 — + reranker", "dev", a3_mean, a3_err, "3 — +reranker"),
        ("Arm 3 + company filter\n+ query strip", "dev", a3f_mean, a3f_err,
         "3 + company filter + query strip"),
        ("Arm 3 + company filter\n+ query strip", "test", t_mean, t_err,
         "3 + filter + strip, TEST"),
    ]

    published = parse_baseline_table()
    problems: list[str] = []
    for _, _, mean, err, row in bars:
        check(published, row, mean, err, problems)
    if problems:
        raise SystemExit(
            "ABORT -- derived numbers disagree with DECISIONS.md's baseline table:\n  "
            + "\n  ".join(problems)
            + "\nOne of the two is wrong. Do not ship a chart until you know which."
        )
    print(f"all {len(bars)} bars agree with DECISIONS.md's current-baseline table")
    return bars, n_dev, n_test


# --- Drawing ------------------------------------------------------------------------------

# Palette: dataviz skill reference instance, light mode. Two categorical slots (1 blue,
# 2 orange); validated with the skill's validate_palette.js: all six checks PASS on surface
# #fcfcfb (worst-pair CVD ΔE 24.7, normal-vision ΔE 33.6).
SURFACE = "#fcfcfb"
DEV_COLOR = "#2a78d6"
TEST_COLOR = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_TERTIARY = "#6f6e6a"
GRID = "#e4e3df"

BAR_HEIGHT = 0.62  # < 1.0 so the surface itself separates neighbouring bars
CORNER_PX = 4.0  # rounded data-end, square at the baseline


def rounded_bar(ax, y: float, width: float, height: float, color: str) -> None:
    """Draw a horizontal bar with square left corners and rounded right (data) corners.

    matplotlib has no per-corner radius, so the bar is an explicit path. The radius is
    given in display pixels and converted per axis, otherwise a wide-and-short axes
    renders the "circle" as a visible ellipse.
    """
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    rx = CORNER_PX * (x1 - x0) / bbox.width
    ry = CORNER_PX * (y1 - y0) / bbox.height
    rx = min(rx, width / 2)
    ry = min(ry, height / 2)

    top, bot = y + height / 2, y - height / 2
    right = width
    verts = [
        (0, bot),
        (right - rx, bot),
        (right, bot),  # control
        (right, bot + ry),
        (right, top - ry),
        (right, top),  # control
        (right - rx, top),
        (0, top),
        (0, bot),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def draw(bars: list[tuple], n_dev: int, n_test: int) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Top-to-bottom in pipeline order. The test bar sits a slot-and-a-half lower so the eye
    # reads it as a separate measurement, not the next step of the progression.
    positions = [4.4, 3.4, 2.4, 1.4, 0.0]

    ax.set_xlim(0, 0.99)
    ax.set_ylim(-0.5, 4.9)

    for pos, (_, split, value, err, _) in zip(positions, bars):
        color = DEV_COLOR if split == "dev" else TEST_COLOR
        rounded_bar(ax, pos, value, BAR_HEIGHT, color)
        ax.errorbar(
            value, pos, xerr=err, fmt="none", ecolor=TEXT_SECONDARY,
            elinewidth=1.2, capsize=3.5, capthick=1.2, zorder=4,
        )
        # Value at the tip, clear of the error-bar cap. Text wears an ink token, never the
        # series colour -- identity comes from the bar beside it.
        ax.text(value + err + 0.018, pos, f"{value:.3f} ± {err:.3f}", va="center", ha="left",
                fontsize=10.5, color=TEXT_PRIMARY,
                fontweight="bold" if split == "test" else "normal")

    # Split is encoded twice -- colour AND the tick label -- so it never rests on hue alone.
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{name}\n({'dev' if split == 'dev' else 'TEST, held out'})" for name, split, *_ in bars],
        fontsize=10, color=TEXT_PRIMARY,
    )
    ax.tick_params(axis="y", length=0, pad=8)

    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="x", length=0, colors=TEXT_SECONDARY, labelsize=9.5)
    ax.set_xlabel("recall@10", fontsize=10, color=TEXT_SECONDARY, labelpad=8)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["left"].set_linewidth(1.0)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9, color=DEV_COLOR,
                   label=f"dev split (n={n_dev:,})"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9, color=TEST_COLOR,
                   label=f"held-out test split (n={n_test:,})"),
    ]
    # Above the plot area: the bars reach the right edge, so no in-axes corner is free.
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.005), ncols=2,
              frameon=False, fontsize=9.5, labelcolor=TEXT_SECONDARY,
              handletextpad=0.5, columnspacing=1.6, borderaxespad=0.0)

    fig.suptitle("Each retrieval stage, measured — recall@10", x=0.013, y=0.975,
                 ha="left", fontsize=14, color=TEXT_PRIMARY, fontweight="bold")
    fig.text(0.013, 0.912,
             "799 SEC 10-K filings · coverage-based gold labels (RETR-35) · post-RETR-7 re-index",
             ha="left", fontsize=8.8, color=TEXT_SECONDARY)
    fig.text(0.013, 0.135,
             "The last two bars are the same configuration on two different question sets, not a "
             "further improvement.",
             ha="left", va="bottom", fontsize=8.2, color=TEXT_SECONDARY)
    # Provenance stamp: what produced this file, from which artifacts, on what day.
    fig.text(0.013, 0.020,
             f"Generated {date.today().isoformat()} by scripts/archive/results_chart.py — every "
             "value recomputed at build time from the artifacts, none typed in.\n"
             "Arms 1–2 aggregated from data/day{3,4}_arm{1,2}_dev_results.json; Arm 3 rows "
             "replayed from data/retr7_rr_{dev,test}_scores.jsonl by\n"
             "05_arm3_rerank.py score. Each value cross-checked against DECISIONS.md's current "
             "baseline table (RETR-39).",
             ha="left", va="bottom", fontsize=7.4, color=TEXT_TERTIARY, linespacing=1.6)

    fig.subplots_adjust(left=0.245, right=0.99, top=0.815, bottom=0.265)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, facecolor=SURFACE)
    print(f"wrote {OUT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--reuse-scored", action="store_true",
        help="skip the two ~90 s replays and read the scorer's committed sidecars "
             "(data/retr7_arm3_{dev,test}_results.json) instead. Layout edits only.",
    )
    args = ap.parse_args()
    bars, n_dev, n_test = collect(args.reuse_scored)
    draw(bars, n_dev, n_test)


if __name__ == "__main__":
    main()
