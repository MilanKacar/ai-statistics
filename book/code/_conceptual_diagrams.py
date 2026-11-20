"""
Internal helper (not a chapter script): generates conceptual/explanatory
diagrams (Venn diagrams, a Bayes natural-frequency tree, a covariance
quadrant plot, a p-value shaded-area plot) that get embedded INLINE next to
the formulas they explain, as opposed to the "run it yourself" simulation
figures which already exist per chapter.
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
# Chapter 3, Figure A: Sample space + complement (NOT), and OR (union)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# (a) Complement
ax = axes[0]
box = patches.Rectangle((-3, -2.2), 6, 4.4, fill=False, edgecolor="#888", linewidth=1.5)
ax.add_patch(box)
ax.text(-2.8, 1.9, r"$\Omega$ (sample space)", fontsize=11, color="#666")
circleA = patches.Circle((0, 0), 1.3, facecolor="#4C78A8", alpha=0.55, edgecolor="#4C78A8", linewidth=2)
ax.add_patch(circleA)
ax.text(0, 0, "A", fontsize=16, ha="center", va="center", color="white", fontweight="bold")
ax.text(2.1, 1.5, "everything\nOUTSIDE A\n= \"not A\"", fontsize=10, ha="center", color="#E45756")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(r"$P(\mathrm{not}\ A) = 1 - P(A)$" + "\nshaded region outside the circle", fontsize=12)

# (b) Union / OR
ax = axes[1]
cA = patches.Circle((-0.7, 0), 1.5, facecolor="#4C78A8", alpha=0.5, edgecolor="#4C78A8", linewidth=2)
cB = patches.Circle((0.7, 0), 1.5, facecolor="#F58518", alpha=0.5, edgecolor="#F58518", linewidth=2)
ax.add_patch(cA); ax.add_patch(cB)
ax.text(-1.3, 0, "A", fontsize=15, ha="center", va="center", fontweight="bold")
ax.text(1.3, 0, "B", fontsize=15, ha="center", va="center", fontweight="bold")
ax.text(0, 0, "A and B\n(counted\ntwice if you\njust add!)", fontsize=8.3, ha="center", va="center",
        color="white", fontweight="bold")
ax.annotate("subtract this\noverlap ONCE", xy=(0, -0.9), xytext=(0, -2.1),
            ha="center", fontsize=10, color="#E45756",
            arrowprops=dict(arrowstyle="->", color="#E45756"))
ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.2); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(r"$P(A\ \mathrm{or}\ B) = P(A)+P(B)-P(A\ \mathrm{and}\ B)$", fontsize=12)

plt.tight_layout()
save(fig, "ch03_venn_not_or.png")

# ---------------------------------------------------------------------------
# Chapter 3, Figure B: AND (intersection), and conditional probability
# (shrinking the sample space)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# (a) Intersection / AND
ax = axes[0]
cA = patches.Circle((-0.7, 0), 1.5, facecolor="#4C78A8", alpha=0.25, edgecolor="#4C78A8", linewidth=2)
cB = patches.Circle((0.7, 0), 1.5, facecolor="#F58518", alpha=0.25, edgecolor="#F58518", linewidth=2)
ax.add_patch(cA); ax.add_patch(cB)
# shade only the intersection using a clip trick: draw a darker circle clipped by both
theta = np.linspace(0, 2 * np.pi, 300)
# approximate intersection lens by sampling points inside both circles
xx, yy = np.meshgrid(np.linspace(-2.6, 2.6, 400), np.linspace(-2.2, 2.2, 400))
inA = (xx + 0.7) ** 2 + yy ** 2 <= 1.5 ** 2
inB = (xx - 0.7) ** 2 + yy ** 2 <= 1.5 ** 2
ax.contourf(xx, yy, (inA & inB).astype(float), levels=[0.5, 1], colors=["#E45756"], alpha=0.75)
ax.text(-1.3, 0, "A", fontsize=15, ha="center", va="center", fontweight="bold")
ax.text(1.3, 0, "B", fontsize=15, ha="center", va="center", fontweight="bold")
ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(r"$P(A\ \mathrm{and}\ B)$" + "\nonly the overlap (red) counts", fontsize=12)

# (b) Conditional probability: shrinking the sample space to B
ax = axes[1]
cA2 = patches.Circle((-0.7, 0), 1.5, facecolor="none", edgecolor="#4C78A8", linewidth=2)
cB2 = patches.Circle((0.7, 0), 1.5, facecolor="none", edgecolor="#F58518", linewidth=3)
ax.add_patch(cA2); ax.add_patch(cB2)
ax.contourf(xx, yy, (inA & inB).astype(float), levels=[0.5, 1], colors=["#E45756"], alpha=0.8)
# dim everything outside B
outsideB = ~inB
ax.contourf(xx, yy, outsideB.astype(float), levels=[0.5, 1], colors=["white"], alpha=0.55)
ax.text(1.3, 1.35, "once we're TOLD\nB happened, B\nbecomes our whole\nnew universe", fontsize=8.8,
        ha="center", color="#F58518", fontweight="bold")
ax.annotate("", xy=(0.3, 0), xytext=(-1.1, 1.0),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))
ax.text(-1.6, 1.15, r"$P(A|B)=\dfrac{P(A\ \mathrm{and}\ B)}{P(B)}$" + "\n= red area / orange circle area",
        fontsize=10, ha="left")
ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.2, 2.4); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Conditional probability:\nshrink the universe down to B", fontsize=12)

plt.tight_layout()
save(fig, "ch03_venn_and_conditional.png")

# ---------------------------------------------------------------------------
# Chapter 4: Bayes' theorem as a natural-frequency tree
# (matches the medical test worked example: prior=1%, sens=99%, spec=99%)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 8))
ax.axis("off")
ax.set_xlim(0, 10.3)
ax.set_ylim(-1.3, 8)


def box(x, y, text, w=2.1, h=0.9, color="#eef1f5", edge="#888"):
    rect = patches.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                   facecolor=color, edgecolor=edge, linewidth=1.3, clip_on=False, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=9.6, zorder=4)


def arrow(x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1 + 0.5), xytext=(x0, y0 - 0.5),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.3))
    if label:
        ax.text((x0 + x1) / 2 + 0.15, (y0 + y1) / 2, label, fontsize=8.6, color="#444")


box(5, 7, "10,000 people\n(population)", color="#dfe7f0")

box(2.3, 5, "100 have the disease\n(1% prior)", color="#f6d9d9")
box(7.7, 5, "9,900 do not\n(99%)", color="#dde8dd")
arrow(5, 7, 2.3, 5, "P(disease)=1%")
arrow(5, 7, 7.7, 5, "P(no disease)=99%")

box(1.1, 2.7, "99 test POSITIVE\n(99% sensitivity)", w=2.2, color="#f2b8b8")
box(3.6, 2.7, "1 tests negative", w=1.9, color="#f6e3e3")
arrow(2.3, 5, 1.1, 2.7, "x0.99")
arrow(2.3, 5, 3.6, 2.7, "x0.01")

box(6.2, 2.7, "99 test POSITIVE\n(1% false alarm)", w=2.2, color="#cfe3cf")
box(8.7, 2.7, "9,801 test negative", w=2.2, color="#dff0df")
arrow(7.7, 5, 6.2, 2.7, "x0.01")
arrow(7.7, 5, 8.7, 2.7, "x0.99")

ax.annotate("", xy=(3.9, 0.1), xytext=(1.1, 2.2), arrowprops=dict(arrowstyle="-", color="#E45756", lw=1.5))
ax.annotate("", xy=(3.9, 0.1), xytext=(6.2, 2.2), arrowprops=dict(arrowstyle="-", color="#E45756", lw=1.5))
box(3.9, -0.05, "198 total positive tests\n= 99 truly sick + 99 false alarms\n\nP(disease | positive) = 99/198 = 50%",
    w=4.8, h=1.15, color="#fff3cd", edge="#E45756")

ax.set_title("Bayes' Theorem as natural frequencies\n(same numbers as the worked medical-test example)",
              fontsize=13)
save(fig, "ch04_bayes_tree.png")

# ---------------------------------------------------------------------------
# Chapter 7: covariance as colored quadrants
# ---------------------------------------------------------------------------
rng = np.random.default_rng(99)
n = 120
x = rng.normal(0, 1, n)
y = 0.6 * x + rng.normal(0, 0.8, n)
xbar, ybar = x.mean(), y.mean()

same_sign = ((x - xbar) * (y - ybar)) > 0

fig, ax = plt.subplots(figsize=(7, 6.2))
ax.scatter(x[same_sign], y[same_sign], color="#4C78A8", label="same sign -> positive product\n(pulls covariance UP)", s=28)
ax.scatter(x[~same_sign], y[~same_sign], color="#E45756", label="opposite sign -> negative product\n(pulls covariance DOWN)", s=28)
ax.axvline(xbar, color="#555", linestyle="--", linewidth=1)
ax.axhline(ybar, color="#555", linestyle="--", linewidth=1)
ax.text(xbar, ax.get_ylim()[1] * 0.92, "  mean of X", fontsize=9, color="#555")
ax.text(ax.get_xlim()[1] * 0.6, ybar, "mean of Y", fontsize=9, color="#555", va="bottom")
ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.set_title("Covariance = average of (X-mean)(Y-mean)\nblue points dominate here, so covariance is positive")
ax.legend(fontsize=8.5, loc="lower right")
plt.tight_layout()
save(fig, "ch07_covariance_quadrants.png")

# ---------------------------------------------------------------------------
# Chapter 12: p-value as shaded tail area(s)
# ---------------------------------------------------------------------------
from scipy import stats

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
x = np.linspace(-4, 4, 400)
y = stats.norm.pdf(x)
z_obs = 2.0

ax = axes[0]
ax.plot(x, y, color="#333", linewidth=1.5)
mask_hi = x >= z_obs
mask_lo = x <= -z_obs
ax.fill_between(x[mask_hi], y[mask_hi], color="#E45756", alpha=0.75)
ax.fill_between(x[mask_lo], y[mask_lo], color="#E45756", alpha=0.75)
ax.axvline(z_obs, color="#4C78A8", linestyle="--")
ax.axvline(-z_obs, color="#4C78A8", linestyle="--")
ax.text(z_obs, max(y) * 0.35, " observed\n z=+2.0", fontsize=9, color="#4C78A8")
p_two = 2 * (1 - stats.norm.cdf(z_obs))
ax.set_title(f"Two-sided p-value\nboth red tails = {p_two:.3f}", fontsize=11)
ax.set_xlabel("z"); ax.set_yticks([])

ax = axes[1]
ax.plot(x, y, color="#333", linewidth=1.5)
ax.fill_between(x[mask_hi], y[mask_hi], color="#E45756", alpha=0.75)
ax.axvline(z_obs, color="#4C78A8", linestyle="--")
ax.text(z_obs, max(y) * 0.35, " observed\n z=+2.0", fontsize=9, color="#4C78A8")
p_one = 1 - stats.norm.cdf(z_obs)
ax.set_title(f"One-sided p-value\nonly the right tail = {p_one:.3f}", fontsize=11)
ax.set_xlabel("z"); ax.set_yticks([])

plt.tight_layout()
save(fig, "ch12_pvalue_shading.png")

print("All conceptual diagrams generated.")
