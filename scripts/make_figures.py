"""Regenerate the report figures without training, model imports, or external data.

From the repository root:
  uv run --no-project --with matplotlib==3.9.2 --with numpy python -I scripts/make_figures.py

All measured values come from docs/data/*.json. Diagram labels describing the
implementation are transcribed from the source audit recorded in REPORT.md.
Run IDs select experiments; no measured results are hard-coded here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
OUT = ROOT / "docs" / "figures"


def read(name):
    return json.loads((DATA / f"{name}.json").read_text())


RUNS = read("runs")
CURVES = read("curves")
TRAIN = read("train_curves")["runs"]
BASELINES = read("baselines")
DATASET = read("dataset_summary")
KEYS = read("keys_openai_meta")
FINAL = ("r16_oai_dense_hist", "r16_oai_dense_hist_seed1")
ARGS = RUNS[FINAL[0]]["args"]

BLUE, ORANGE, TEAL, PURPLE = "#2266A3", "#C76725", "#137D79", "#8455A4"
INK, GRAY, PALE = "#233448", "#637181", "#F1F5F8"
COLORS = (BLUE, ORANGE, TEAL, PURPLE)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": GRAY,
    "ytick.color": GRAY, "axes.edgecolor": "#CDD5DE",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "svg.fonttype": "none",
    "svg.hashsalt": "workflow-embeddings-report", "legend.frameon": False,
})

# Explicit, configuration-matched seed pairs used in SUMMARY.md. Missing old
# flags mean their disabled defaults; True prev_query is the later "sum" mode.
SEED_PAIRS = [
    ("r2_crop_paraneg_reg", "r3_reg_seed1"),
    ("r3_small_sd30", "r4_small_sd30_seed1"),
    ("r4_small_sd30_k50", "r5_seed1"),
    ("r6_combo", "r6_combo_seed1"),
    ("r6_combo_ls10", "r7_seed1"),
    ("r7_reg", "r8_reg_seed1"),
    ("r8_prevq", "r8_prevq_seed1"),
    ("r8_prevq_reg_ls20", "r9_best_seed1"),
    ("r9_sep", "r9_sep_seed1"),
    ("r10_keep2", "r11_keep2_seed1"),
    ("r11b_effect", "r11_effect_seed1"),
    ("r12_mix4", "r12_mix4_seed1"),
    ("r14_dense_hist", "r15_dense_hist_seed1"),
    ("r14_dense_perc32_hist", "r14_dense_perc32_hist_seed1"),
    ("r14_nocrop_hist", "r15_nocrop_hist_seed1"),
    ("r16_oai_dense_hist", "r16_oai_dense_hist_seed1"),
    ("r16_oai_nocrop_hist", "r16_oai_nocrop_hist_seed1"),
]

LABEL_PAIRS = [
    ("8×8 · seed 0", "r14_nocrop_hist", "r16_oai_nocrop_hist"),
    ("8×8 · seed 1", "r15_nocrop_hist_seed1", "r16_oai_nocrop_hist_seed1"),
    ("16×16 · seed 0", "r14_dense_hist", "r16_oai_dense_hist"),
    ("16×16 · seed 1", "r15_dense_hist_seed1", "r16_oai_dense_hist_seed1"),
]


def metrics(run, split):
    """Corrected reevaluations take precedence; never substitute last checkpoints."""
    r = RUNS[run]
    return r.get(f"reeval_{split}") or r.get(f"final_{split}") or (
        r["best_val"] if split == "val" else {})


def judged(run, metric, split="test"):
    return metrics(run, split)[f"jsession_{metric}_cross_app"]


def noise(split):
    return float(np.median([abs(metrics(a, split)["loss"] - metrics(b, split)["loss"])
                            for a, b in SEED_PAIRS]))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.15,
                metadata={"Date": None})
    plt.close(fig)
    print(f"{name}.png / .svg")


def chart(ax, ylabel=None, xlabel=None):
    ax.grid(axis="y", color="#E5EAF0", linewidth=0.7)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def diagram(w=14, h=6, xmax=100, ymax=100):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set(xlim=(0, xmax), ylim=(0, ymax))
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, label, edge=BLUE, face=PALE, size=11, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1",
                               facecolor=face, edgecolor=edge, linewidth=1.25))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size,
            fontweight=weight, linespacing=1.45)


def arrow(ax, start, end, color=GRAY, style="-", rad=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.35, color=color, linestyle=style,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3))


def overview():
    fig, ax = diagram(h=6.3)
    ax.text(2, 97, "A session is evidence arriving over time", fontsize=15, weight="bold")
    for x, t in [(5, "1"), (36, "t"), (67, "T")]:
        box(ax, x, 76, 24, 13, f"Screenshot {t}\n+ input events {t}", size=11)
        box(ax, x, 52, 24, 14, "Recurrent state machine\nshared weights", edge=TEAL)
        arrow(ax, (x + 12, 76), (x + 12, 66))
        arrow(ax, (x + 12, 52), (x + 12, 45))
        ax.text(x + 12, 40, f"direction d$_{{{t}}}$ + confidence κ$_{{{t}}}$",
                ha="center", fontsize=10)
    arrow(ax, (29, 59), (36, 59), TEAL)
    arrow(ax, (60, 59), (67, 59), TEAL)
    ax.text(32.5, 64, "state", ha="center", color=TEAL, fontsize=9)
    ax.text(63.5, 64, "state", ha="center", color=TEAL, fontsize=9)
    ax.text(2, 59, "h₀", ha="center", color=TEAL)
    arrow(ax, (2, 59), (5, 59), TEAL)
    box(ax, 3, 4, 55, 23,
        f"TRAINING: supervise every retained step\n"
        f"Frozen embeddings of all {DATASET['splits']['train']:,} training instructions\n"
        "κ × cosine scores → contrastive loss", edge=ORANGE, face="#FFF5EA")
    for x in (17, 48, 79):
        arrow(ax, (x, 36), (min(x, 52), 27), ORANGE, "--")
    box(ax, 63, 4, 34, 23,
        "INFERENCE: screenshots + events\nCompare two sessions by\ncosine(d$_T$, d′$_T$)",
        edge=TEAL, face="#EBF6F4")
    ax.text(50, -2, "d is a unit vector; κ scales instruction scores. Instruction text is a training target.",
            ha="center", fontsize=10, color=GRAY)
    save(fig, "01_approach")


def architecture():
    fig, ax = diagram(w=15, h=9.2, xmax=120, ymax=115)
    d, h = ARGS["d"], ARGS["h"]
    ax.text(1, 112, "One step of the final model", fontsize=16, weight="bold")
    box(ax, 2, 88, 34, 18,
        "Frozen SigLIP2 whole screen\n16×16 grid · 1,152 channels\nchannel PCA → 256×512", edge=BLUE)
    box(ax, 41, 88, 39, 18,
        "Event encoder → aₜ ∈ ℝ³⁸⁴\ntype + Fourier coords + scroll + keys\nQwen3-Embedding-0.6B typed text", edge=ORANGE, size=10.5)
    box(ax, 88, 94, 29, 12, "Previous state\nhₜ₋₁ ∈ ℝ⁵¹²", edge=TEAL)
    box(ax, 2, 62, 34, 18,
        "LayerNorm → linear K, V\nlearned position added to K\n256×384 each; token drop 0.4", size=10.5)
    arrow(ax, (19, 88), (19, 80))
    box(ax, 41, 60, 31, 20,
        "8 action queries × 384\nlearned base + Wₐaₜ\n+ Wₚaₜ₋₁ (sum mode)", edge=ORANGE, size=10.5)
    box(ax, 76, 60, 18, 20, "4 static\nlearned queries\n4×384", edge=PURPLE, size=10.5)
    box(ax, 98, 60, 19, 20, "4 history queries\nWₕhₜ₋₁\n4×384", edge=TEAL, size=10)
    arrow(ax, (57, 88), (57, 80), ORANGE)
    arrow(ax, (103, 94), (107, 80), TEAL)
    ax.text(76, 84, "aₜ₋₁ is stored", color=ORANGE, fontsize=9, ha="center")
    box(ax, 40, 37, 55, 14,
        "Cross-attention: 16 queries, 8 heads × 48 channels\n"
        "concatenate 16 outputs (6,144) → linear → fₜ (384)", size=11)
    arrow(ax, (19, 62), (40, 44))
    for x in (57, 85, 107):
        arrow(ax, (x, 60), (min(x, 91), 51))
    box(ax, 4, 12, 45, 15,
        "Fusion [fₜ ; aₜ] (768)\nLinear 768→768 · GELU · 768→384\nDropout 0.3 → zₜ", edge=ORANGE, size=10.5)
    arrow(ax, (53, 37), (33, 27))
    # Event branch reaches fusion directly as well as generating visual queries.
    ax.plot([41, 38.5, 38.5, 2, 2], [97, 97, 33, 33, 20], color=ORANGE, lw=1.2)   # runs in the gap left of the attention box
    arrow(ax, (2, 20), (4, 20), ORANGE)
    ax.text(14, 34, "aₜ bypass", fontsize=9, color=ORANGE)
    box(ax, 58, 12, 42, 15,
        f"GRU cell: input {d}, state {h}\nLayerNorm → hₜ (512)\nlearned h₀; freeze state on padding", edge=TEAL, size=10.5)
    arrow(ax, (49, 19), (58, 19))
    ax.plot([117.5, 119, 119, 105], [100, 100, 19, 19], color=TEAL, lw=1.2)
    arrow(ax, (105, 19), (100, 19), TEAL)
    arrow(ax, (79, 12), (79, 4), TEAL)
    ax.text(79, 0, "Training dropout 0.3 before both heads", ha="center", fontsize=10)
    ax.text(25, -8, f"dₜ = normalize(Linear 512→{KEYS['dim']:,})", ha="center", fontsize=12, color=BLUE)
    ax.text(88, -8, "κₜ = min(50, 1 + softplus(Linear 512→1))", ha="center", fontsize=11.5, color=TEAL)
    ax.set_ylim(-13, 115)
    save(fig, "02_architecture")


def objective():
    fig, ax = diagram(w=14, h=6.3)
    ax.text(2, 96, "One session, one retained step: all training instructions are candidates", fontsize=14, weight="bold")
    ax.text(2, 91, f"Frozen normalized keys: OpenAI {KEYS['model']} ({KEYS['dim']:,} dimensions)",
            fontsize=10, color=GRAY)
    box(ax, 3, 73, 25, 15, "Model emits dₜ, κₜ\nlogit j = κₜ dₜ · u′ⱼ", edge=TEAL)
    box(ax, 34, 73, 63, 15,
        "For EVERY candidate column, draw one wording uniformly\noriginal instruction or a retained paraphrase; redraw each optimizer step",
        edge=ORANGE, size=10.5)
    arrow(ax, (17, 73), (17, 61), TEAL)
    arrow(ax, (65, 73), (65, 61), ORANGE)
    cells = [(3, "other", BLUE), (19, "true target", TEAL), (35, "same group", GRAY),
             (51, "cosine > 0.8", GRAY), (67, "similar", BLUE), (83, "other", BLUE)]
    for x, label, color in cells:
        masked = color == GRAY
        box(ax, x, 42, 14, 19, label + ("\n−∞ (masked)" if masked else "\nκₜ dₜ · u′ⱼ"),
            edge=color, face="#ECEFF2" if masked else PALE, size=10)
        if masked:
            ax.plot([x + 1, x + 13], [43, 60], color="#B6C0CA", linewidth=1)
        ax.text(x + 7, 35, "q = 0" if masked else ("q = 0.8" if label == "true target" else "share of 0.2"),
                ha="center", fontsize=10, color=color)
    ax.text(50, 25,
            "Mask and smoothing use ORIGINAL keys u, not the sampled wordings u′. The target is never masked.",
            ha="center", fontsize=10)
    box(ax, 3, 2, 46, 17,
        "Other unmasked columns share 0.2\nin proportion to exp(cos(uᵢ, uⱼ) / 0.05)\nexclude the target from this distribution",
        edge=ORANGE, size=10.5)
    box(ax, 54, 2, 43, 17,
        "Softmax over unmasked logits → −Σⱼ qⱼ log pⱼ\nAverage steps within each session,\nthen average sessions", edge=TEAL, size=10.5)
    ax.text(50, -5, f"Schematic columns only; actual training has {DATASET['splits']['train']:,} candidate columns. No measured logits are depicted.",
            ha="center", color=GRAY, fontsize=10)
    save(fig, "03_objective")


def learning():
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.7), layout="constrained")
    for seed, run in enumerate(FINAL):
        color = COLORS[seed]
        tr = np.array(TRAIN[run]["train"], float)
        va = np.array(TRAIN[run]["val"], float)
        for ax in axes[1:]:
            ax.plot(va[:, 0], va[:, 1], color=color, lw=1.8,
                    ls="-" if seed == 0 else "--", label=f"Validation · seed {seed}")
            ax.plot(RUNS[run]["best_step"], metrics(run, "val")["loss"], "*", color=color,
                    ms=11, markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        axes[0].plot(tr[:, 0], tr[:, 1], color=color, lw=1.4,
                     ls="-" if seed == 0 else "--", label=f"Seed {seed}")
        test = RUNS[run]["final_test"]["loss"]
        axes[2].plot(RUNS[run]["best_step"], test, "x" if seed == 0 else "+", color=color,
                     ms=9, mew=1.7, zorder=6)
        axes[2].annotate(f"Test seed {seed}: {test:.3f}",
                         xy=(RUNS[run]["best_step"], test),
                         xytext=(7600, 4.82 - seed * 0.20), fontsize=9, color=color,
                         arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8})
    axes[0].set_title("A  Logged training objective")
    axes[0].legend(loc="upper right")
    axes[0].text(.97, .52, "50-step running mean\nTraining augmentations active\nDifferent loss scale from evaluation",
                 transform=axes[0].transAxes, ha="right", fontsize=9, color=GRAY)
    axes[1].set_title("B  Full validation trajectory")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[2].set_title("C  Validation after warm-up (zoom)")
    axes[2].set(xlim=(3000, 13000), ylim=(3.5, 5.04))
    axes[2].axvline(RUNS[FINAL[0]]["best_step"], color=GRAY, ls=":", lw=1)
    axes[2].text(5700, 3.52, f"★ selected: {RUNS[FINAL[0]]['best_step']:,}", fontsize=9, va="bottom")
    axes[2].text(.97, .89, "Test: two checkpoint measurements\nNo test curve was recorded", ha="right",
                 transform=axes[2].transAxes, fontsize=9, color=GRAY)
    for i, ax in enumerate(axes):
        chart(ax, "Training objective" if i == 0 else "Evaluation loss (nats)", "Optimizer step")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    save(fig, "04_learning_curves")


def development():
    milestones = [
        ("R1\nOriginal", "r1_v1"),
        ("R1\nNo paraphrases", "r1_nopara"),
        ("R2\nSymmetric targets\n+ regularization", "r2_crop_paraneg_reg"),
        ("R3\nSmaller state\n+ step dropout", "r3_small_sd30"),
        ("R4\nκ cap 50", "r4_small_sd30_k50"),
        ("R6\nTraining bundle\n+ smoothing", "r6_combo_ls10"),
        ("R8\nPrevious action\n+ stronger regularization", "r8_prevq_reg_ls20"),
        ("R14\nRemove crops", "r14_nocrop"),
        ("R14\nDense + history\n+ static queries", "r14_dense_hist"),
        ("R15\n8 history queries", "r15_dense_hist8"),
    ]
    fig, ax = plt.subplots(figsize=(14.4, 5.6), layout="constrained")
    for split, color in [("val", BLUE), ("test", ORANGE)]:
        y = [metrics(n, split)["loss"] for _, n in milestones]
        ax.plot(range(len(y)), y, color=color, lw=1.3, alpha=.8)
        for x, ((_, run), val) in enumerate(zip(milestones, y)):
            crop = RUNS[run]["args"]["use_crop"]
            ax.plot(x, val, "o" if crop else "s", ms=7, color=color,
                    markerfacecolor=color if crop else "white", markeredgewidth=1.6)
            ax.annotate(f"{val:.2f}", (x, val), xytext=(0, 9 if split == "test" else -16),
                        textcoords="offset points", ha="center", fontsize=9, color=color)
    ax.axvline(6.5, color=GRAY, ls=":", lw=1)
    ax.text(6.65, 7.35, "No-crop constraint\nA6000 → A100 also changed", fontsize=9, color=GRAY)
    ax.set_xticks(range(len(milestones)), [x for x, _ in milestones], fontsize=9)
    ax.set_ylim(3.78, 7.65)
    ax.set_title("Selected Qwen-label milestones: loss at each run's best-validation checkpoint", loc="left")
    chart(ax, "Evaluation loss (nats; lower is better)")
    ax.legend(handles=[Line2D([], [], color=BLUE, label="Validation"),
                       Line2D([], [], color=ORANGE, label="Test"),
                       Line2D([], [], color=GRAY, marker="o", ls="", label="With crops"),
                       Line2D([], [], color=GRAY, marker="s", mfc="white", ls="", label="Without crops")],
              ncol=4, loc="upper right", bbox_to_anchor=(1, .83), fontsize=9)
    save(fig, "08_development")


# Each row is treatment minus control at matched seed, never a cumulative gain.
ABLATIONS = [
    ("Symmetric paraphrases vs none †", [("r2_crop_nopara", "r2_crop_paraneg")]),
    ("Step dropout alone, larger model †", [("r2_crop_paraneg_reg", "r3_sd30")]),
    ("Smaller model + step dropout", [("r2_crop_paraneg_reg", "r3_small_sd30"),
                                      ("r3_reg_seed1", "r4_small_sd30_seed1")]),
    ("κ cap: 100 → 50", [("r3_small_sd30", "r4_small_sd30_k50"),
                          ("r4_small_sd30_seed1", "r5_seed1")]),
    ("Training bundle (includes EMA)", [("r4_small_sd30_k50", "r6_combo"),
                                       ("r5_seed1", "r6_combo_seed1")]),
    ("Similarity smoothing: 0 → 0.1", [("r6_combo", "r6_combo_ls10"),
                                      ("r6_combo_seed1", "r7_seed1")]),
    ("Add previous-action queries", [("r6_combo_ls10", "r8_prevq"),
                                     ("r7_seed1", "r8_prevq_seed1")]),
    ("Dense: direct vs fixed averaging †", [("r14_dense_avg8", "r14_dense_direct")]),
    ("Dense: history + static queries †", [("r14_dense_direct", "r14_dense_hist")]),
    ("Dense + history: learned compression", [("r14_dense_hist", "r14_dense_perc32_hist"),
                                             ("r15_dense_hist_seed1", "r14_dense_perc32_hist_seed1")]),
]


def ablation():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True, layout="constrained")
    for ax, split in zip(axes, ("val", "test")):
        n = noise(split)
        ax.axvspan(-n, n, color="#DAE1E8", alpha=.8, zorder=0)
        ax.axvline(0, color=GRAY, lw=1)
        for y, (_, pairs) in enumerate(ABLATIONS):
            for i, (a, b) in enumerate(pairs):
                delta = metrics(b, split)["loss"] - metrics(a, split)["loss"]
                yy = y + ((i - .5) * .18 if len(pairs) > 1 else 0)
                ax.plot(delta, yy, "o" if i == 0 else "s", color=COLORS[i], ms=7,
                        markeredgecolor="white", markeredgewidth=.6, zorder=3)
        ax.set_title(f"{split.title()} loss: treatment − control")
        ax.set_xlabel("← improvement                 worsening →")
        ax.set_xlim(-.51, .46)
        ax.set_yticks(range(len(ABLATIONS)), [a for a, _ in ABLATIONS], fontsize=10)
        ax.grid(axis="y", color="#E5EAF0", linewidth=.7)
        ax.text(.02, .02, f"Shaded band: ±{n:.3f}\nmedian |seed difference|, {len(SEED_PAIRS)} pairs",
                transform=ax.transAxes, fontsize=9, color=GRAY)
    axes[0].set_ylim(len(ABLATIONS) + .9, -.6)
    axes[1].legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="", label="Seed 0"),
                            Line2D([], [], color=ORANGE, marker="s", ls="", label="Seed 1")],
                   loc="lower right")
    fig.suptitle("What mattered: matched Qwen-label comparisons († single-seed control and treatment)",
                 fontsize=13, fontweight="bold")
    save(fig, "09_ablations")


def uncertainty():
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), layout="constrained")
    run = FINAL[0]  # Only this final seed has per-position arrays in curves.json.
    for split, color in [("val", BLUE), ("test", ORANGE)]:
        c = CURVES[run][split]
        ent = np.asarray(c["entropy_vs_relpos"])
        x = (np.arange(len(ent)) + .5) / len(ent)
        axes[0].plot(x, ent, "o-", color=color, label=split.title(), ms=4)
        axes[1].plot(x, np.asarray(c["true_key_R@5_vs_relpos"]) * 100, "o-", color=color,
                     label=f"{split.title()}: κ first {c['kappa_first']:.2f} → last {c['kappa_last']:.2f}", ms=4)
    axes[0].set_title("A  Distribution over instruction keys")
    axes[1].set_title("B  True instruction among the top five")
    axes[0].legend()
    axes[1].legend(loc="lower right", fontsize=9)
    for ax in axes:
        ax.set_xlim(0, 1)
        chart(ax, xlabel=f"Relative session position ({len(ent)} bins; plotted at bin centers)")
    axes[0].set_ylabel("Entropy (nats)")
    axes[1].set_ylabel("True-key R@5 (%)")
    axes[1].set_ylim(0, 78)
    fig.suptitle("Final model, seed 0: evidence accumulates while κ is already close to its cap", fontsize=13, weight="bold")
    save(fig, "07_uncertainty")


def label_space():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.7), layout="constrained")
    for ax, metric in zip(axes, ("R@1", "R@5", "MRR", "nDCG@10")):
        deltas = []
        for color, (label, qwen, oai) in zip(COLORS, LABEL_PAIRS):
            a, b = 100 * judged(qwen, metric), 100 * judged(oai, metric)
            ax.plot([0, 1], [a, b], "o-", color=color, lw=1.6, ms=5, label=label)
            deltas.append(b - a)
        ax.set_xticks([0, 1], ["Qwen", "OpenAI"])
        ax.set_xlim(-.25, 1.25)
        ax.set_title(f"{metric}\nmean Δ {np.mean(deltas):+.2f} points")
        chart(ax, "Score × 100")
        ax.margins(y=.22)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=10)
    fig.suptitle("Judged test retrieval: same architecture and seed, different instruction targets", fontsize=13, weight="bold")
    save(fig, "05_label_space")


def baselines():
    labels = ["Action TF-IDF", "Mean-frame features", "Last-frame features", "Qwen text-key oracle", "Final model (2 seeds)"]
    names = ["action_tfidf", "mean_frame", "last_frame", "text_key_oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True, layout="constrained")
    for ax, metric in zip(axes, ("R@5", "nDCG@10")):
        vals = [BASELINES[f"val/{n}"][f"jsession_{metric}_cross_app"] for n in names]
        final = [judged(n, metric, "val") for n in FINAL]
        vals.append(float(np.mean(final)))
        ax.barh(range(len(vals)), np.array(vals) * 100,
                color=["#A6BACD"] * 3 + ["#A991BB", TEAL], height=.58)
        ax.errorbar(100 * np.mean(final), 4,
                    xerr=[[100 * (np.mean(final) - min(final))], [100 * (max(final) - np.mean(final))]],
                    fmt="none", ecolor=INK, capsize=4)
        for i, v in enumerate(vals):
            text_x = (max(final) if i == len(vals) - 1 else v) * 100 + 2
            ax.text(text_x, i, f"{v * 100:.1f}", va="center", fontsize=10)
        ax.set_title(f"Validation judged {metric}")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlim(0, 106)
        ax.set_xlabel("Score × 100")
        ax.grid(axis="x", color="#E5EAF0", lw=.7)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    nq = metrics(FINAL[0], "val")["n_jsession_queries_cross_app"]
    fig.suptitle(f"Small validation pool: {nq} same-workflow queries; oracle uses instruction text", fontsize=13, weight="bold")
    save(fig, "06_baselines")


def memorization():
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), layout="constrained")
    c = np.array(CURVES["r1_nopara"]["kappa_first_over_training"])
    for ax, col, color, title, ylabel in [
        (axes[0], 2, BLUE, "A  Validation starts to worsen", "Validation loss (nats)"),
        (axes[1], 1, ORANGE, "B  First-step confidence keeps rising", "Mean first-step κ")]:
        ax.plot(c[:, 0], c[:, col], "o-", color=color, ms=4)
        best = c[np.argmin(c[:, 2])]
        ax.axvline(best[0], color=GRAY, lw=1, ls=":")
        for row in (best, c[-1]):
            ax.annotate(f"{row[col]:.3f}" if col == 2 else f"{row[col]:.2f}", (row[0], row[col]), xytext=(-8, 10),
                        textcoords="offset points", ha="right", color=color)
        ax.set_title(title)
        chart(ax, ylabel, "Optimizer step")
        ax.margins(y=.2)
    fig.suptitle("Early no-paraphrase run: a diagnostic of overfitting, not proof of a specific shortcut", fontsize=13, weight="bold")
    save(fig, "10_memorization")


def main():
    for fn in (overview, architecture, objective, learning, development, ablation,
               uncertainty, label_space, baselines, memorization):
        fn()
    print(f"Seed-noise bands: val={noise('val'):.6f}; test={noise('test'):.6f}")


if __name__ == "__main__":
    main()
