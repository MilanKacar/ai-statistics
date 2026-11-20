# Chapter 15: Multiple Regression, R², and Overfitting

## The idea this chapter builds toward

Real-world outcomes rarely depend on just one input variable. **Multiple regression** extends Chapter 14's single-predictor line into a model with many predictors at once, and in doing so, exposes a genuinely dangerous failure mode hiding inside the least-squares machinery: adding more predictors *always* makes your fit look better on the data you trained on, even when those predictors are pure noise. Understanding exactly why that happens — not just that it happens — is this chapter's real payoff, because it is the single clearest, simplest example of **overfitting**, the central practical problem in all of machine learning.

## The model, extended to many predictors

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon$$

Each $\beta_j$ now represents the effect of $x_j$ **holding all other predictors constant** — a crucial and non-obvious interpretation shift from simple regression. If house price depends on both square footage and number of bedrooms, and those two predictors are themselves correlated (bigger houses tend to have more bedrooms), then $\beta_{\text{bedrooms}}$ answers "what's the effect of one more bedroom, for houses of the *same* square footage" — not the naive, uncontrolled correlation between bedrooms and price alone. This "holding everything else fixed" interpretation is both multiple regression's main power and its main source of confusion (it requires caring about which other variables are in the model, since adding or removing predictors can change a coefficient's size or even its sign — a famous example of this is discussed below as **Simpson's Paradox**).

### Deriving the fit — the same recipe, one dimension up

The least-squares logic from Chapter 14 generalizes directly: minimize $SSE = \sum_i (y_i - \hat y_i)^2$ where now $\hat y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}$. With more than one predictor, doing this derivative-sets-to-zero process by hand for each $\beta_j$ becomes unwieldy with plain algebra, which is exactly the problem that motivates Chapter 19's linear algebra — writing the whole system compactly as matrices lets you solve for *all* the coefficients simultaneously with a single matrix equation, $\hat\beta = (X^TX)^{-1}X^Ty$, which we'll build up to properly once you have the matrix vocabulary. For now, trust that it's doing the exact same "minimize sum of squared residuals" job as Chapter 14, just handling many predictors at once.

## Simpson's Paradox — why "holding other variables constant" really matters

A famous real case: UC Berkeley's 1973 graduate admissions showed men were admitted at a notably higher overall rate than women — suggestive of bias. But when admissions were broken down *department by department*, women were admitted at an equal or *higher* rate than men in most individual departments. The resolution: women disproportionately applied to more competitive departments (with lower acceptance rates for everyone), while men disproportionately applied to less competitive ones — department choice was a **confounding variable** that reversed the apparent relationship once properly accounted for. This is exactly why multiple regression — explicitly controlling for a confound like department by including it as a predictor — is a core tool for avoiding exactly this kind of misleading aggregate conclusion, and why "correlation isn't causation" (Chapter 7) so often specifically needs multiple regression as its practical fix, not just a verbal caveat.

## R² always goes up when you add predictors — here's exactly why

This is the crux of the chapter, so follow the mechanism closely. $SSE$ (the leftover squared error after fitting) can mathematically never *increase* when you add another predictor to the model — because the least-squares fitting procedure is *free* to set that new predictor's coefficient to exactly zero if it's genuinely useless, which would reproduce the old model's fit exactly. In practice, with real (noisy, finite) data, the optimizer will almost never land on an *exact* zero coefficient — it will find some small nonzero coefficient that fits the *specific noise* in your particular training sample slightly better, shaving $SSE$ down by some small amount purely by chance, even for a predictor that's genuinely unrelated to $y$ in the true underlying population. Since $R^2 = 1 - SSE/SST$ and $SST$ doesn't change, a shrinking $SSE$ mechanically forces $R^2$ to rise (or at worst stay flat) with every additional predictor — **even predictors you generate from pure random noise.** This is not a bug or an edge case; it is a mathematical guarantee baked directly into how least-squares optimization works, and it means **$R^2$ alone can never tell you whether an added predictor is genuinely useful or whether you're just fitting noise** — the code below demonstrates this with literal random-noise columns added to a real model, and you'll watch $R^2$ climb anyway.

## Adjusted R² — a first attempt at a fix

$$R^2_{\text{adj}} = 1 - (1-R^2)\frac{n-1}{n-p-1}$$

**Intuition**: this explicitly penalizes the raw $R^2$ for the number of predictors $p$ used, relative to the number of data points $n$. As $p$ grows (more predictors, for fixed $n$), the $\frac{n-1}{n-p-1}$ correction factor grows, pulling $R^2_{\text{adj}}$ down — unlike plain $R^2$, adjusted $R^2$ *can* decrease when you add a genuinely useless predictor, because the complexity penalty outweighs the small, noise-driven improvement in raw fit. This is your first explicit encounter with a recurring theme that dominates the rest of this book: **a good model-quality metric must explicitly trade off "fits the data well" against "isn't needlessly complex,"** or you will systematically be fooled into preferring overly complicated models. Adjusted R² is a fairly blunt, specific instrument for this; Chapter 16 generalizes the exact same idea (the **bias-variance tradeoff**) into the central organizing principle of practical machine learning, and Chapter 22 formalizes it even further into statistical learning theory.

## Overfitting, stated plainly for the first time

**Overfitting** is what you just watched happen mechanically: a model that fits its *training data* better and better (ever-rising raw $R^2$) while actually getting *worse* at the thing you actually care about — making good predictions on *new, unseen* data. This is precisely the population-vs-sample distinction from Chapter 1, now showing real teeth: a model can memorize idiosyncratic noise specific to your particular training sample, mistaking that noise for genuine signal about the broader population, and that memorized noise actively hurts predictions on any *different* sample drawn from the same population. This single idea — distinguishing "fits the data I have" from "will generalize to data I don't have" — is arguably the single most important practical concept in all of applied machine learning, and Chapter 16 gives it the full, rigorous treatment it deserves.

## Figures

![Raw R^2 keeps climbing as more and more USELESS random-noise predictors are added to the model, while adjusted R^2 stays flat or declines -- overfitting made mechanically visible.](../images/ch15_multiple_regression_fig1.png)

*Raw R^2 keeps climbing as more and more USELESS random-noise predictors are added to the model, while adjusted R^2 stays flat or declines -- overfitting made mechanically visible.*

## Run it yourself

`book/code/ch15_multiple_regression.py` fits a multiple regression model on a realistic synthetic housing dataset (square footage + bedrooms + age, predicting price), then deliberately adds an increasing number of pure-random-noise predictor columns and plots how raw $R^2$ climbs steadily upward even though those columns are provably unrelated to the target — while adjusted $R^2$ stays flat or declines, giving you a direct, visual demonstration of overfitting's mechanical cause before Chapter 16 names and formalizes it properly.

```bash
python book/code/ch15_multiple_regression.py
```

## Why This Matters — In AI and In Everyday Life

**In AI / machine learning:**

- **Feature engineering pitfalls**: adding many engineered features to a model without checking adjusted R² or using cross-validation is a direct, common real cause of production models that looked great in development and failed in the real world.
- **Controlling for confounds in ML fairness analysis**: checking whether a model's predictions differ by a protected attribute *after* controlling for legitimate factors is a direct real application of multiple regression's "holding other variables constant" logic.

**In everyday, non-AI life:**

- **Predicting salary from experience, education, and location together**: a classic HR analytics use of multiple regression, where the Simpson's Paradox risk from this chapter is a genuine, frequently-made mistake.
- **UC Berkeley admissions (Simpson's Paradox)**: the real 1973 case in this chapter is itself a profoundly real-life example of why controlling for confounders matters in any fairness-sensitive analysis, not just a toy illustration.

## Check your understanding

- Explain, in your own words, why least-squares optimization *mechanically guarantees* that $SSE$ cannot increase when a new predictor is added, even a useless one.
- Why does adjusted R² sometimes decrease when raw R² increases, and what does that tell you about the tradeoff being penalized?
- In the Simpson's Paradox example, what role does "department" play, and why does leaving it out of the analysis distort the apparent gender-admission relationship?

## What's next

Chapter 16 takes the overfitting phenomenon you just watched happen and gives it a rigorous mathematical decomposition — the **bias-variance tradeoff** — along with the standard practical tools (cross-validation, regularization) used throughout machine learning to detect and control it.
