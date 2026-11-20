# Chapter 9: Sampling and Estimation

## The idea this chapter builds toward

An **estimator** is a *rule/formula* you apply to sample data to guess at an unknown population parameter (Chapter 1's $\mu$ vs $\bar{x}$ distinction, now made rigorous). Not all estimators are equally good — this chapter gives you precise language (bias, variance, consistency, efficiency) for comparing them, which matters because an entire area of statistics and ML (from "why divide by $n-1$" in Chapter 2, to "why regularize" in Chapter 16) is really arguments about which estimator to trust.

## Estimator vs. estimate — a small but important distinction

An **estimator** is the general *procedure* (e.g., "take the sample average" — a formula, $\hat\mu = \bar X = \frac1n\sum X_i$). An **estimate** is the specific *number* you get when you apply that procedure to one particular dataset (e.g., "32.25," from Chapter 1's example). The hat symbol ($\hat\theta$, "theta-hat") is standard notation for "an estimator or estimate of the true parameter $\theta$."

Because an estimator is a formula applied to *random* sample data, the estimator itself is a **random variable** — it has its own distribution (its "sampling distribution"), exactly what Chapter 8's CLT described for the sample mean specifically.

## What makes an estimator "good"? Three properties

### 1. Bias

$$\text{Bias}(\hat\theta) = E[\hat\theta] - \theta$$

**Intuition**: if you ran your estimation procedure over and over on many different samples, would the *average* of all those estimates converge to the true value, or would it systematically overshoot/undershoot? An estimator is **unbiased** if $E[\hat\theta] = \theta$ exactly — no systematic error, even though any *single* estimate will still be off due to sampling randomness.

This is exactly why Chapter 2 divides sample variance by $n-1$ instead of $n$: dividing by plain $n$ produces an estimator whose *average* value across many samples is systematically a bit too small (biased low) — dividing by $n-1$ corrects that bias to exactly zero. You can now see "Bessel's correction" for what it really is: a bias-correction derived directly from this chapter's definition.

### 2. Variance (of the estimator)

How much does the estimate bounce around from sample to sample (exactly what Chapter 1's visualization showed, and what Chapter 8's standard error, $\sigma/\sqrt n$, quantifies for the sample mean specifically). Low variance means your estimates are consistent/stable across different samples, even if (hypothetically) biased.

### 3. The bias-variance tradeoff — a first preview

You generally cannot minimize bias and variance simultaneously beyond a point — pushing one down can push the other up. A silly but illustrative extreme estimator: "always guess the number 50, no matter what data you see." This has zero variance (always the same answer!) but is almost certainly badly biased (unless the truth happens to be exactly 50). Meanwhile $\bar X$ has zero bias but nonzero variance. Which is "better" depends on the relative sizes of the two errors — a question we'll return to with real teeth in Chapter 16, where this exact tradeoff explains overfitting vs underfitting in machine learning models.

![Four dartboards illustrating low/high bias crossed with low/high variance, with the bullseye representing the true value being estimated.](../images/ch09_bias_variance_dartboard.png)

*The classic picture for this tradeoff: the bullseye is the true value you're trying to hit. **Bias** is whether your throws are centered on the bullseye on average (top row vs. bottom row). **Variance** is how scattered your throws are around wherever they're centered (left column vs. right column). Top-left is what every estimator wants to be; the other three panels are the specific, distinct ways an estimator can fail — and they require different fixes, which is exactly why "it's a little off" is never a complete diagnosis without knowing whether the problem is bias, variance, or both.*

### 4. Consistency

An estimator is **consistent** if, as sample size $n \to \infty$, the estimate converges to the true value with probability approaching 1 (i.e., with enough data, you're essentially guaranteed to get arbitrarily close to the truth). $\bar X$ is consistent — this is a direct consequence of the CLT's shrinking standard error ($\sigma/\sqrt n \to 0$ as $n \to \infty$) combined with being unbiased.

## Good vs. bad estimators: a concrete comparison

Suppose you want to estimate the "center" of a dataset. Candidate estimators: the sample mean, the sample median, or "just the first data point you happened to collect."

- **Sample mean**: unbiased for a symmetric distribution, low variance when $n$ is large (shrinks like $1/\sqrt n$), consistent. Generally a strong default.
- **Sample median**: also often unbiased for symmetric distributions, slightly *higher* variance than the mean when data is close to Normal, but far more **robust** to outliers (recall Chapter 2's income example) — so the "better" choice depends on whether your data has heavy outliers.
- **"First data point you collected"**: technically unbiased (its expected value equals the population mean, since it's a single random draw from the population)! But its variance is $\sigma^2$ — the *full* population variance, not shrunk at all by averaging, and it does **not** improve as you collect more data (collecting more data just means you're ignoring most of it). This is **inconsistent** in practice despite technically being unbiased — a good illustration that "unbiased" alone is not enough to make an estimator good; you need low variance (or consistency) too.

## The standard error, generalized

Chapter 8 gave you the standard error of the mean specifically: $SE(\bar X) = \sigma/\sqrt n$. More generally, the **standard error of an estimator** is just the standard deviation of its sampling distribution — "how much does this specific estimator bounce around across different samples of this size." Every estimator has its own standard error formula; the mean's happens to be the clean $\sigma/\sqrt n$ one because of the CLT, but other estimators (like the median, or a regression coefficient in Chapter 14) have their own, sometimes more complex, standard error formulas. The *concept* — "quantify how much this number would change if we re-sampled" — is universal and is exactly what feeds into confidence intervals (Chapter 11).

## The backstory — Ronald Fisher turns "estimation" into a rigorous science

Before the 1920s, statisticians used estimators somewhat informally — "take the average," "take the middle value" — without a crisp, general theory for comparing *why* one choice might be better than another. **Ronald A. Fisher**, working largely in the 1910s-20s (notably while analyzing decades of agricultural crop-yield data at the Rothamsted Experimental Station in England — the same work that leads directly into ANOVA in Chapter 13), set out to formalize exactly this comparison problem. He introduced much of the vocabulary you just learned — consistency, efficiency, and especially **sufficiency** (a related idea: does an estimator retain *all* the useful information in the data, or does it throw some away?) — as part of building statistics into a mathematically rigorous discipline rather than a bag of ad hoc recipes.

Fisher's motivating puzzle, worth sitting with because it's a genuinely good intuition-builder: given two different *unbiased* estimators for the same parameter, how do you decide which is "better" if both are correct on average? His answer was variance — of two unbiased estimators, prefer the one that varies *less* across hypothetical repeated samples (this is literally the definition of **efficiency**: an "efficient" estimator is the lowest-variance one among the unbiased options). This is exactly why, when choosing between the sample mean and sample median as an estimator of a Normal distribution's center (both unbiased for symmetric data), statisticians default to the mean — it has strictly lower variance for Normal data specifically (though, as the "income" example in Chapter 2 hints, the median's extra variance is often a worthwhile price to pay for robustness against outliers in non-Normal, real-world data — a tradeoff Fisher's clean theoretical framework doesn't automatically resolve for you, and still requires human judgment about your actual data).

## Why this matters for ML directly

Every number a model outputs or reports — a trained weight, a measured test accuracy, an estimated feature importance — is an *estimate* with its own bias and variance, even though ML practice often forgets to report it. "This model gets 94% accuracy" is itself an estimate of the model's *true* accuracy on the full population of possible inputs, with its own standard error depending on test-set size — reporting a bare accuracy number without any sense of its uncertainty is exactly the mistake this chapter is designed to prevent you from making unknowingly.

## Figures

![Sampling distributions of three estimators (sample mean, sample median, first data point) across three sample sizes. The red dashed line marks the true population mean -- compare how tightly each estimator clusters around it.](../images/ch09_sampling_and_estimation_fig1.png)

*Sampling distributions of three estimators (sample mean, sample median, first data point) across three sample sizes. The red dashed line marks the true population mean -- compare how tightly each estimator clusters around it.*

## Run it yourself

`book/code/ch09_sampling_and_estimation.py` compares the three "center" estimators above (mean, median, first-data-point) across many repeated samples from a known population, computing each estimator's empirical bias and variance directly, and visualizing how each one's sampling distribution narrows (or doesn't) as sample size grows.

```bash
python book/code/ch09_sampling_and_estimation.py
```

## Why This Matters — In AI and In Everyday Life

**In AI / machine learning:**

- **Judging a reported model metric**: "91% accuracy" is an estimator's output with its own bias and variance -- this chapter is exactly what should make you ask what sample size and sampling procedure produced that number.
- **Biased training data as a biased estimator**: if your labeled training data was collected in a way that systematically over- or under-represents certain cases, your model's learned parameters inherit that bias, in the precise sense defined in this chapter.

**In everyday, non-AI life:**

- **Choosing between average and median home price**: realtors and the median-vs-mean choice directly reflects which estimator (Chapter 2's robustness argument, Chapter 9's variance/bias tradeoff) better represents "typical."
- **Government economic statistics**: unemployment rate and inflation figures are estimators computed from surveys, each with a known margin of error that government statisticians report alongside the headline number.

## Check your understanding

- Explain why "always guess 50" has zero variance but is (usually) badly biased, and why "sample mean" is the reverse tradeoff.
- Why is the "first data point" estimator unbiased yet still a bad choice in practice?
- If someone reports "our model's accuracy is 91%" with no other information, what question from this chapter should you immediately ask?

## What's next

Chapter 10 introduces **Maximum Likelihood Estimation** — a single, general *principle* for constructing good estimators (rather than guessing formulas like "take the average" ad hoc), which turns out to be the exact same mathematical object as the "loss function" you minimize when training almost any machine learning model.
