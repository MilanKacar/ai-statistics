"""
Internal helper (not a chapter script): second batch of conceptual/intuition
diagrams - the classic "aha" pictures for beginners:
  - why squaring deviations makes variance work (literal squares = area)
  - the bias-variance dartboard
  - ANOVA's between-group vs within-group variance
  - the EM algorithm as a two-step cycle
  - a Markov chain state diagram
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(IMAGES_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------------------
# Chapter 2: why squaring deviations works -- literal squares = area
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

data = np.array([2, 4, 5, 8, 9])
m = data.mean()

ax = axes[0]
ax.axhline(0, color="#888", linewidth=1)
ax.scatter(data, np.zeros_like(data), s=60, color="#4C78A8", zorder=5)
ax.axvline(m, color="#E45756", linestyle="--", label=f"mean={m:.1f}")
for x in data:
    ax.annotate("", xy=(m, 0.35), xytext=(x, 0.35),
                arrowprops=dict(arrowstyle="->", color="#F58518", lw=1.4))
    ax.text((x+m)/2, 0.42, f"{x-m:+.1f}", fontsize=8.5, ha="center", color="#F58518")
ax.set_ylim(-0.3, 0.9)
ax.set_xlim(0, 11)
ax.set_yticks([])
ax.set_xlabel("value")
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Raw deviations from the mean\n(always sum to exactly zero)", fontsize=11)

ax = axes[1]
ax.axhline(0, color="#888", linewidth=1)
ax.scatter(data, np.zeros_like(data), s=60, color="#4C78A8", zorder=5)
ax.axvline(m, color="#E45756", linestyle="--")
for x in data:
    side = abs(x - m)
    if side < 0.05:
        continue
    rect = patches.Rectangle((min(x, m), 0), side, side, facecolor="#54A24B", alpha=0.45, edgecolor="#54A24B")
    ax.add_patch(rect)
    ax.text(min(x, m) + side/2, side/2, f"{side:.1f}²", fontsize=8, ha="center", va="center")
ax.set_xlim(0, 11)
ax.set_ylim(-0.3, 7.5)
ax.set_xlabel("value")
ax.set_title("Each deviation SQUARED = literal area of a square\n(always positive, bigger misses cost much more)", fontsize=11)

plt.tight_layout()
save(fig, "ch02_squared_deviations.png")

# ---------------------------------------------------------------------------
# Chapter 9 / 16: the bias-variance dartboard
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
rng = np.random.default_rng(5)

configs = [
    ("Low bias, low variance\n(what you want)", 0, 0.05, "#54A24B"),
    ("Low bias, high variance\n(overfitting-ish: noisy but centered)", 0, 0.35, "#F58518"),
    ("High bias, low variance\n(underfitting: consistent but off-target)", 0.55, 0.05, "#E45756"),
    ("High bias, high variance\n(worst of both)", 0.55, 0.35, "#888888"),
]

for ax, (title, bias, spread, color) in zip(axes.flat, configs):
    # draw dartboard rings
    for r, c in zip([1.0, 0.7, 0.4, 0.15], ["#f3f3f3", "#dde8f0", "#c3d7e8", "#a9c6e0"]):
        ax.add_patch(patches.Circle((0, 0), r, facecolor=c, edgecolor="#999", linewidth=0.8, zorder=1))
    ax.add_patch(patches.Circle((0, 0), 0.04, facecolor="#4C78A8", edgecolor="none", zorder=1))

    pts_x = rng.normal(bias, spread, 18)
    pts_y = rng.normal(0, spread, 18)
    ax.scatter(pts_x, pts_y, color=color, s=35, zorder=5, edgecolor="white", linewidth=0.5, clip_on=False)

    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10.5)

plt.suptitle("The bias-variance dartboard\nbullseye = the true value you're trying to estimate or predict", fontsize=13)
plt.tight_layout()
save(fig, "ch09_bias_variance_dartboard.png")

# ---------------------------------------------------------------------------
# Chapter 13: ANOVA -- between-group vs within-group variance
# ---------------------------------------------------------------------------
rng = np.random.default_rng(11)
groupA = rng.normal(70, 4, 12)
groupB = rng.normal(78, 4, 12)
groupC = rng.normal(85, 4, 12)
groups = [groupA, groupB, groupC]
grand_mean = np.concatenate(groups).mean()

fig, ax = plt.subplots(figsize=(8.5, 6))
positions = [1, 2, 3]
colors = ["#4C78A8", "#F58518", "#54A24B"]
ax.set_xlim(0.0, 4.1)

for pos, g, c in zip(positions, groups, colors):
    ax.scatter(np.full_like(g, pos) + rng.normal(0, 0.03, len(g)), g, color=c, alpha=0.7, s=28, zorder=5, clip_on=False)
    ax.plot([pos-0.22, pos+0.22], [g.mean(), g.mean()], color=c, linewidth=3, clip_on=False)
    ax.annotate("", xy=(pos+0.35, g.mean()), xytext=(pos+0.35, grand_mean),
                arrowprops=dict(arrowstyle="<->", color=c, lw=1.3), annotation_clip=False)

ax.axhline(grand_mean, color="#888", linestyle="--", linewidth=1.3, label=f"grand mean = {grand_mean:.1f}")

# within-group spread illustration for group A
gx = positions[0]
ax.annotate("", xy=(gx-0.4, groupA.max()), xytext=(gx-0.4, groupA.min()),
            arrowprops=dict(arrowstyle="<->", color="#333", lw=1.1), annotation_clip=False)
ax.text(gx-0.62, groupA.mean(), "within-\ngroup\nspread", fontsize=8, ha="center", color="#333")

ax.set_xticks(positions); ax.set_xticklabels(["Method A", "Method B", "Method C"])
ax.set_ylabel("test score")
ax.legend(fontsize=9, loc="lower right")
ax.set_title("ANOVA's two variances:\ncolored double-arrows = between-group gaps from the grand mean\nblack double-arrow = within-group noise", fontsize=11)
plt.tight_layout()
save(fig, "ch13_anova_between_within.png")

# ---------------------------------------------------------------------------
# Chapter 23: EM algorithm as a two-step cycle
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 8)

def box(x, y, text, w=3.6, h=1.5, color="#eef1f5"):
    rect = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                   facecolor=color, edgecolor="#555", linewidth=1.4)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5)

box(2.6, 5.6, "E-step\nGiven current (mu, sigma, pi),\ncompute each point's probability\nof belonging to each cluster", color="#dbe7f3")
box(7.4, 5.6, "M-step\nGiven those soft assignments,\nre-estimate (mu, sigma, pi)\nvia weighted MLE", color="#f6e3d0")
box(5, 1.8, "Repeat until the parameters\nstop changing (converged)\n\nLikelihood never decreases\nat any step", w=4.6, h=1.7, color="#fdf3cf")

ax.annotate("", xy=(5.3, 5.6), xytext=(4.4, 5.6), arrowprops=dict(arrowstyle="->", color="#333", lw=2))
ax.annotate("", xy=(7.4, 4.0), xytext=(7.4, 4.85), arrowprops=dict(arrowstyle="->", color="#333", lw=2))
ax.annotate("", xy=(2.3, 4.75), xytext=(3.7, 2.9),
            arrowprops=dict(arrowstyle="->", color="#333", lw=2, connectionstyle="arc3,rad=-0.35"),
            annotation_clip=False)
ax.text(1.5, 3.6, "loop back", fontsize=8.5, color="#333", style="italic")

ax.set_title("The EM algorithm: a two-step loop", fontsize=13)
plt.tight_layout()
save(fig, "ch23_em_cycle.png")

# ---------------------------------------------------------------------------
# Chapter 24: Markov chain state diagram (Sunny/Rainy weather example)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("off")
ax.set_xlim(-1, 11); ax.set_ylim(-1, 8)

r = 1.2
sun = (2.2, 3.5)
rain = (8.2, 3.5)
ax.add_patch(patches.Circle(sun, r, facecolor="#fde9b0", edgecolor="#e8b324", linewidth=2, zorder=3))
ax.add_patch(patches.Circle(rain, r, facecolor="#c9dcf0", edgecolor="#4C78A8", linewidth=2, zorder=3))
ax.text(*sun, "Sunny", ha="center", va="center", fontsize=14, fontweight="bold", zorder=4)
ax.text(*rain, "Rainy", ha="center", va="center", fontsize=14, fontweight="bold", zorder=4)

# transition arrow Sunny -> Rainy (curves above the straight line)
arrow1 = patches.FancyArrowPatch(
    (sun[0] + r*0.75, sun[1] + r*0.65), (rain[0] - r*0.75, rain[1] + r*0.65),
    connectionstyle="arc3,rad=0.35", arrowstyle="-|>", mutation_scale=18,
    color="#333", linewidth=1.8, zorder=2)
ax.add_patch(arrow1)
ax.text((sun[0]+rain[0])/2, 6.1, "P(Rainy | Sunny) = 0.2", fontsize=11, ha="center")

# transition arrow Rainy -> Sunny (curves below the straight line)
arrow2 = patches.FancyArrowPatch(
    (rain[0] - r*0.75, rain[1] - r*0.65), (sun[0] + r*0.75, sun[1] - r*0.65),
    connectionstyle="arc3,rad=0.35", arrowstyle="-|>", mutation_scale=18,
    color="#333", linewidth=1.8, zorder=2)
ax.add_patch(arrow2)
ax.text((sun[0]+rain[0])/2, 0.85, "P(Sunny | Rainy) = 0.4", fontsize=11, ha="center")

# self-loop Sunny (prob 0.8) -- small loop above the circle
loop1 = patches.FancyArrowPatch(
    (sun[0] - r*0.4, sun[1] + r*1.05), (sun[0] + r*0.4, sun[1] + r*1.05),
    connectionstyle="arc3,rad=2.3", arrowstyle="-|>", mutation_scale=16,
    color="#c9951a", linewidth=1.8, zorder=6)
ax.add_patch(loop1)
ax.text(sun[0], sun[1] + r*2.0, "P(Sunny | Sunny) = 0.8", fontsize=10.5, ha="center", color="#c9951a")

# self-loop Rainy (prob 0.6)
loop2 = patches.FancyArrowPatch(
    (rain[0] - r*0.4, rain[1] + r*1.05), (rain[0] + r*0.4, rain[1] + r*1.05),
    connectionstyle="arc3,rad=2.3", arrowstyle="-|>", mutation_scale=16,
    color="#4C78A8", linewidth=1.8, zorder=6)
ax.add_patch(loop2)
ax.text(rain[0], rain[1] + r*2.0, "P(Rainy | Rainy) = 0.6", fontsize=10.5, ha="center", color="#4C78A8")

ax.set_title("Markov chain state diagram:\nthe future depends only on today's state, not on history", fontsize=13)
plt.tight_layout()
save(fig, "ch24_markov_state_diagram.png")

print("Second batch of conceptual diagrams generated.")
