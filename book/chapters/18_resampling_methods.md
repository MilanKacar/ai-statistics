# Chapter 18: Resampling — Bootstrap and Permutation Tests

## The idea this chapter builds toward

Chapters 11-13's confidence intervals and hypothesis tests leaned on known formulas (standard error $=\sigma/\sqrt n$, t-distributions, chi-square distributions) that depend on specific distributional assumptions (Normality, independence, known sample-size behavior). **Resampling methods** sidestep needing those formulas at all: instead of deriving a sampling distribution mathematically, you *simulate* it directly from your own data, repeatedly, by a computer. This chapter gives you two such methods — the **bootstrap** (for building confidence intervals around almost any statistic) and **permutation tests** (for hypothesis testing without distributional assumptions) — both of which became practical only once computers made brute-force repeated resampling cheap, making this one of the more recent major ideas in this book's timeline.

## The bootstrap — the core trick, explained from first principles

Here's the conceptual leap, and it's worth sitting with because it initially sounds almost like cheating: you don't have access to the true population (Chapter 1's whole problem), so you can't draw new samples from it to see how your statistic (say, the median) would vary from sample to sample. The bootstrap's trick: **treat your one actual sample as a stand-in for the population**, and draw new "fake" samples *from it*, **with replacement**, of the same size as your original sample. Because your sample is itself (hopefully) a reasonably representative slice of the real population, resampling from it mimics — imperfectly, but often remarkably well — what would happen if you could actually draw fresh samples from the true population.

### Why "with replacement" specifically?

If you resampled *without* replacement at the same size $n$ as your original sample, you'd just get back the exact same sample every single time (same $n$ items, no replacement, from a pool of exactly $n$ items — there's only one way to do that). Sampling *with replacement* means each "fake" resample can include some original data points multiple times and omit others entirely — introducing exactly the kind of sample-to-sample variability that a real repeated-sampling-from-the-population process would also produce, letting you empirically observe how your statistic fluctuates.

### The procedure, step by step

1. From your original sample of size $n$, draw a new sample of size $n$, **with replacement** (some original points repeat, some are left out).
2. Compute your statistic of interest (mean, median, correlation, a regression coefficient — anything) on this resampled "bootstrap sample."
3. Repeat steps 1-2 many times (typically 1,000-10,000 times), building up a whole distribution of the statistic across all these bootstrap resamples.
4. This distribution of bootstrap statistics *is* your empirical approximation of the statistic's sampling distribution — directly usable to build a confidence interval (e.g., take the 2.5th and 97.5th percentiles of the bootstrap distribution for a 95% interval — called the **percentile method**), with no formula for standard error ever required.

### Why this is especially useful beyond the mean

Chapter 11's confidence interval formula worked specifically for the mean, leaning on the CLT. But what if you want a confidence interval for the **median**, or for a **correlation coefficient**, or for the **ratio of two medians** — statistics with no clean, universally-agreed-upon standard-error formula (or one that requires strong, possibly-wrong distributional assumptions)? The bootstrap handles *any* statistic identically, with the same simple resample-and-recompute procedure — this generality, not raw statistical power, is its primary appeal.

## The backstory — Bradley Efron names and formalizes an old gambling-adjacent idea

**Bradley Efron** introduced the bootstrap formally in a 1979 paper, explicitly naming it after the phrase "pull yourself up by your bootstraps" — an idiom for accomplishing something seemingly impossible using only what you already have on hand (originally a sarcastic, physically-impossible image, used here in a knowingly self-deprecating way to describe a method that sounds almost too good to be true: estimating a sampling distribution's spread without any new data or any distributional formula). Efron's genuinely sharp insight was recognizing that a specific family of older, more narrowly-scoped resampling techniques (notably Quenouille and Tukey's earlier "jackknife" method, which works by systematically leaving out one observation at a time rather than resampling with replacement) could be generalized into one clean, universal, and remarkably broadly-applicable procedure — and that the then-newly-cheap availability of computing power made running thousands of resamples for *any* dataset entirely practical for the first time, unlocking a method whose core logic had arguably existed in pieces for decades but had never been computationally affordable enough to use routinely.

## Permutation tests — hypothesis testing by shuffling labels

Where the bootstrap builds confidence intervals, **permutation tests** answer a hypothesis-testing question (Chapter 12) with a similarly simulation-based, assumption-light approach. Suppose you're testing whether two groups (e.g., a treatment and control group) have genuinely different means. The null hypothesis is that group *labels are meaningless* — that treatment and control come from the same underlying distribution, and any observed difference is just due to which data points randomly happened to land in which group.

**The trick**: if the null hypothesis is true, the group labels really are arbitrary — so you can randomly **shuffle** (permute) which data points are labeled "treatment" vs "control," recompute the difference in group means for this shuffled/fake labeling, and repeat this shuffling thousands of times. This builds up an empirical distribution of "what differences in means look like when the labels are meaningless" — directly from your actual data, with zero distributional assumptions. Your p-value is then simply: *what fraction of these shuffled, label-meaningless differences were as extreme as (or more extreme than) the difference you actually observed in the real, correctly-labeled data?* If your real observed difference is far more extreme than almost all the shuffled versions, that's strong evidence the group labels genuinely matter.

This method traces back conceptually to **Ronald Fisher** himself (the same Fisher from Chapters 9, 10, and 13) in the 1930s, who described permutation-based reasoning as the logically "exact" foundation underlying his more commonly-used parametric tests (like the t-test) — but doing it by hand for even modest sample sizes was computationally prohibitive before computers (the number of possible permutations explodes combinatorially with sample size, e.g. splitting just 20 subjects into two groups of 10 already has over 184,000 distinct permutations) — another example, alongside the bootstrap and MCMC (Chapters 17 and 24), of an idea that existed in theory long before computers made it practically usable.

## Worked example, conceptually

Two groups of website visitors: 15 saw design A, converting at particular rates; 15 saw design B. Observed difference in conversion rate: 4 percentage points, in B's favor. A permutation test would: combine all 30 visitors' outcomes into one pool, repeatedly reshuffle which 15 get relabeled "A" and which 15 get relabeled "B" (ignoring the true original labels), and recompute the fake "difference" each time. If only, say, 2% of these 10,000 random reshufflings produce a fake difference as large as the real observed 4 points, your permutation p-value is 0.02 — strong evidence the real design difference isn't just random noise in who happened to land in which group.

## Figures

![The bootstrap distribution of the sample median, built from 10,000 resamples with replacement, with the 95% percentile-based confidence interval marked.](../images/ch18_resampling_methods_fig1.png)

*The bootstrap distribution of the sample median, built from 10,000 resamples with replacement, with the 95% percentile-based confidence interval marked.*

![A permutation test's null distribution (differences in conversion rate under randomly shuffled group labels), with the real observed difference marked in red.](../images/ch18_resampling_methods_fig2.png)

*A permutation test's null distribution (differences in conversion rate under randomly shuffled group labels), with the real observed difference marked in red.*

## Run it yourself

`book/code/ch18_resampling_methods.py` builds a bootstrap confidence interval for the **median** of a skewed dataset (a statistic with no simple textbook standard-error formula) and compares it to a naive Normal-approximation interval to show where they diverge, then runs a full permutation test on a two-group conversion-rate comparison, visualizing the simulated null distribution of shuffled differences against the real observed difference.

```bash
python book/code/ch18_resampling_methods.py
```

## Why This Matters — In AI and In Everyday Life

**In AI / machine learning:**

- **Bootstrap confidence intervals for any ML metric**: when there's no clean textbook formula for a metric's uncertainty (e.g. an F1 score or a custom business metric), bootstrapping the test set is the standard practical fix used in real ML evaluation.
- **Permutation-based feature importance**: a standard, assumption-light way to measure how much a feature matters to a trained model is to permute (shuffle) that feature's values and see how much performance drops -- literally this chapter's permutation-test logic repurposed.

**In everyday, non-AI life:**

- **Estimating uncertainty in a small-sample medical or industrial study**: when you only have a handful of measurements and no clean formula applies, bootstrapping lets you still get a defensible confidence interval.
- **Testing whether a process change actually helped**, using only the limited before/after data a small factory or clinic actually has, rather than assuming a textbook distribution that may not fit.

## Check your understanding

- Why does resampling *with* replacement (not without) introduce the variability needed to approximate a sampling distribution?
- In your own words, explain what the "null world" looks like in a permutation test, and why shuffling labels simulates it directly.
- For what kind of statistic would you reach for the bootstrap instead of a classical formula-based confidence interval, and why?

## What's next

Chapter 19 takes a short but essential detour into **linear algebra** — vectors, matrices, and the geometric intuition behind them — which you'll need as genuine prerequisite vocabulary for multivariate statistics (Chapter 20), the compact matrix form of multiple regression promised back in Chapter 15, and essentially all of modern machine learning notation from here onward.
