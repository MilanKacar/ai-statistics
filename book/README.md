# Statistics for AI — A Book From Zero to Deep Learning Foundations

This is a self-contained course on the statistics you need to deeply understand machine learning and AI — written for someone who is **rusty on math but comfortable with Python**. Every chapter leads with intuition, builds up the formal idea slowly, works a concrete numeric example by hand, and then gives you a small Python script to run so you can *see* the concept, not just read about it.

## How the book is organized

- `chapters/` — one Markdown file per chapter. Read them in order; later chapters assume earlier ones.
- `code/` — one Python script per chapter (same number), referenced at the point in the chapter where it's useful. Run them locally and play with the numbers.

## Setup (do this once)

You need Python 3.9+. From the project root (one level above `book/`), create a virtual environment and install everything from `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then to run any chapter's code, e.g. chapter 2:

```bash
python book/code/ch02_descriptive_stats.py
```

Each script will print output to the console and/or pop up a matplotlib plot window. If you're in an environment without a display (e.g. a remote server), add this to the top of any script to save the plot to a file instead of showing it:

```python
import matplotlib
matplotlib.use("Agg")  # before importing pyplot
```

and change `plt.show()` to `plt.savefig("output.png")`.

## Reading the book as a real, live website — one command

```bash
python3 book/main.py
```

This builds the site and starts serving it, then opens **http://localhost:8000/** in your browser automatically. Press `Ctrl+C` to stop it. Run it again any time you've edited a chapter — it rebuilds automatically before serving.

Useful flags: `--no-build` (just re-serve the existing build), `--port 8080` (use a different port), `--no-browser` (don't auto-open a tab).

What you get that a raw `.md` file in an editor can't show you:
- A **persistent sidebar** with the full table of contents (grouped by part), so you can jump to any chapter from anywhere — never just prev/next.
- **Figures embedded directly in the text**, right where they're discussed (plus an optional scroll-synced thumbnail rail on wide screens — it only adds a shortcut, it never hides anything, and it disappears entirely on mobile).
- **Live, interactive modules built directly into the relevant chapters** — not a separate page, but a slider-driven simulation sitting right in the paragraph that explains the concept. For example: Chapter 1 lets you draw your own samples and watch the sample mean bounce around; Chapter 4 has a live Bayes' theorem calculator; Chapter 8 lets you watch the Central Limit Theorem turn any population shape into a bell curve; Chapter 16 lets you drag a polynomial's degree and watch overfitting happen in real time. Just scroll to the relevant section and play with it.
- **Highlighting**: select any text → a floating toolbar lets you mark it 🟡 interesting / 🟢 understood / 🔴 confusing / 🔵 review later, or 📝 attach a written note. Click any highlight to edit or remove it.
- A **📒 My Notes** panel (top-right) listing every highlight/note, with jump-to links, plus **Export/Import** to back up your notes or carry them to another device (everything is saved in your browser's local storage — static site, no backend, so there's no automatic sync between devices).
- Formulas rendered properly via MathJax.

(Why not just `Ctrl+Shift+V` in VS Code? That renders Markdown and images fine, but deliberately blocks `<script>` execution for security — so the sidebar, highlighting, and every live interactive module would sit inert. `main.py` builds real, independent `.html` files that any browser runs with no restrictions.)

## Table of Contents

### Part 0 — Orientation
- [Chapter 0: How to Use This Book, and the Math Notation You'll See](chapters/00_how_to_use_this_book.md)

### Part I — Describing the World: Data and Probability
- [Chapter 1: Data, Populations, and Samples — Thinking Like a Statistician](chapters/01_data_populations_samples.md)
- [Chapter 2: Describing Data — Center, Spread, and Shape](chapters/02_descriptive_statistics.md)
- [Chapter 3: Probability Foundations — Events, Rules, and Conditioning](chapters/03_probability_foundations.md)
- [Chapter 4: Bayes' Theorem — Updating Beliefs With Evidence](chapters/04_bayes_theorem.md)
- [Chapter 5: Random Variables and Discrete Distributions](chapters/05_random_variables_discrete.md)
- [Chapter 6: Continuous Distributions and the Normal Distribution](chapters/06_continuous_distributions.md)
- [Chapter 7: Expectation, Variance, Covariance, and Correlation](chapters/07_expectation_variance_covariance.md)
- [Chapter 8: The Central Limit Theorem — The Secret Engine of Statistics](chapters/08_central_limit_theorem.md)

### Part II — Inferring From Data
- [Chapter 9: Sampling and Estimation](chapters/09_sampling_and_estimation.md)
- [Chapter 10: Maximum Likelihood Estimation](chapters/10_maximum_likelihood_estimation.md)
- [Chapter 11: Confidence Intervals](chapters/11_confidence_intervals.md)
- [Chapter 12: Hypothesis Testing and p-values](chapters/12_hypothesis_testing.md)
- [Chapter 13: t-tests, Chi-square, and ANOVA](chapters/13_ttests_chisquare_anova.md)
- [Chapter 14: Correlation and Simple Linear Regression](chapters/14_simple_linear_regression.md)
- [Chapter 15: Multiple Regression, R², and Overfitting](chapters/15_multiple_regression.md)

### Part III — The Bridge Into Machine Learning
- [Chapter 16: Bias-Variance Tradeoff, Cross-Validation, and Regularization](chapters/16_bias_variance_regularization.md)
- [Chapter 17: Bayesian Statistics in Depth](chapters/17_bayesian_statistics.md)
- [Chapter 18: Resampling — Bootstrap and Permutation Tests](chapters/18_resampling_methods.md)
- [Chapter 19: Linear Algebra Essentials for Statistics and ML](chapters/19_linear_algebra_essentials.md)
- [Chapter 20: Multivariate Statistics — Covariance Matrices and PCA](chapters/20_multivariate_pca.md)

### Part IV — The Statistics Inside Modern AI
- [Chapter 21: Information Theory — Entropy, Cross-Entropy, and KL Divergence](chapters/21_information_theory.md)
- [Chapter 22: Statistical Learning Theory — Why Generalization Works At All](chapters/22_statistical_learning_theory.md)
- [Chapter 23: Probabilistic Models and Latent Variables (Mixtures, EM)](chapters/23_latent_variable_models.md)
- [Chapter 24: Markov Chains and Monte Carlo Methods](chapters/24_markov_chains_monte_carlo.md)
- [Chapter 25: The Statistics Hiding Inside Neural Networks](chapters/25_statistics_in_neural_networks.md)
- [Chapter 26: Where to Go Next](chapters/26_where_to_go_next.md)

### Part V — The Statistical Frontier of Modern AI
- [Chapter 27: The Statistical Frontier — How Modern AI Actually Uses These Ideas Today](chapters/27_frontier_ai_statistics.md) — best-of-N sampling, reward models, the winner's curse (reward hacking), self-consistency, and test-time compute scaling.

## How to actually learn this (not just read it)

1. Read a chapter once, slowly, on paper-math parts — don't skip the "by hand" examples even though they're tedious. That's where intuition is built.
2. Run the chapter's code. Change the numbers. Break it. See what changes.
3. Before moving to the next chapter, try to explain the current chapter's core idea out loud, in plain words, with no formulas. If you can't, re-read.

Progress, don't rush. Every later chapter is building on the intuition from an earlier one.
