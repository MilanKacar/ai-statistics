# Chapter 26: Where to Go Next

## What you've actually built

Looking back across this book: you started with the raw idea that any number you compute from data is a guess about something bigger (Chapter 1), built up the full machinery of probability (Chapters 3-8), learned to quantify and manage uncertainty rigorously (Chapters 9-13), built genuine predictive models and learned exactly how and why they can fool you (Chapters 14-16), met the Bayesian alternative to the whole frequentist framework (Chapter 17), picked up the computational and linear-algebra tools modern statistics and ML actually run on (Chapters 18-20), and finished by explicitly tracing every one of those ideas directly into the internals of how neural networks are built and trained (Chapters 21-25). That is a genuinely complete, non-superficial foundation - not a tour of buzzwords, but the actual mathematical and historical reasoning underneath them. From here, the book intentionally stops rather than trying to cover everything; what follows is a map of the most natural next directions, each one build directly on something you already have.

> **Intuition, in one breath:** You now own the whole toolkit - the remaining work is not learning new tools but choosing which real problem to point them at next. Every direction from here is just one of these familiar tools pushed further.

## Causal inference - going beyond "correlation isn't causation"

This book has repeatedly flagged correlation-vs-causation as a trap (Chapter 7's ice cream/drowning example, Simpson's Paradox in Chapter 15) without giving you the formal tools to resolve it. **Causal inference** is the field that does exactly that - formalizing *when* you can legitimately conclude "X causes Y" from observational data (not just a designed experiment), using tools like **directed acyclic graphs (DAGs)** to explicitly represent assumed causal structure, the **do-calculus** (Judea Pearl's framework, distinguishing $P(Y\mid X)$ from the subtly different $P(Y\mid \text{do}(X))$ - "what Y looks like when you *observe* X" vs. "what Y looks like when you *intervene* and force X"), and techniques like instrumental variables, propensity score matching, and difference-in-differences for extracting causal effects from messy real-world data where a clean randomized experiment isn't available. This is the natural next stop if you work with observational data and need genuinely defensible "what if we changed X" claims, not just "X and Y are related" claims.

## Time series analysis - when your data points aren't independent

Nearly every method in this book assumed independent observations (Chapter 3's independence assumption underlies almost everything downstream). Real sequential data - stock prices, sensor readings, website traffic over time - routinely violates this: today's value is correlated with yesterday's. **Time series analysis** builds the specific tools for this setting: autocorrelation (correlation of a series with a lagged version of itself - a direct extension of Chapter 7's correlation), classical models like ARIMA (combining autoregression, differencing for trend removal, and moving averages), and the specific statistical tests for stationarity (whether a series's statistical properties stay constant over time) that most time-series methods require as a prerequisite assumption - directly analogous to how Chapter 14's regression required checking the homoscedasticity assumption via residual plots.

## Experimental design and A/B testing at scale

Chapter 12 gave you the mechanics of a single hypothesis test; real production A/B testing systems need to handle: sequential testing (making valid decisions as data streams in continuously, rather than waiting for one fixed, pre-planned sample size - naively peeking at results repeatedly as they arrive reintroduces exactly Chapter 12's multiple-comparisons problem, and specialized sequential testing methods exist specifically to control for this), multi-armed bandits (dynamically shifting traffic toward better-performing variants *during* the experiment, trading off exploration vs. exploitation - a genuinely different mathematical framework from classical fixed-sample hypothesis testing), and proper handling of network effects and interference between test subjects (when one user's experience isn't fully independent of another's, violating Chapter 3's independence assumption in a different way than time series does). Fisher's experimental design principles from Chapter 13 (randomization, blocking) remain the conceptual bedrock underneath all of this modern machinery.

## Deepening any single chapter further

If one particular chapter's topic especially grabbed you, here's where that thread typically leads next, concretely: Chapter 16/22 (bias-variance, learning theory) → PAC learning theory and generalization bounds for specific modern architectures. Chapter 17/24 (Bayesian stats, MCMC) → probabilistic programming (Stan, PyMC, NumPyro) and Bayesian deep learning. Chapter 20 (PCA) → other dimensionality reduction methods like t-SNE and UMAP, which handle nonlinear structure PCA's linear-only machinery can't. Chapter 21 (information theory) → the broader field of information theory itself, including channel capacity and coding theory, Shannon's original 1948 concerns. Chapter 23 (latent variables) → variational autoencoders and the broader modern generative modeling literature, including diffusion models, whose training objectives are, true to this book's recurring theme, ultimately variations on likelihood-based and KL-divergence-based losses dressed in new architectural clothing.

## Worked example - by hand, larger

Here is a small end-to-end calculation that touches several of the directions above at once, so you can see how the book's tools chain together. Suppose an A/B test of a new recommendation click-through rate (CTR) shows: control group has 2000 users with 240 clicks (12.0%), and treatment group has 2500 users with 340 clicks (13.6%). You want to decide whether to ship it.

**Step 1 - pooled proportion.** Total clicks $= 240+340 = 580$; total users $= 2000+2500 = 4500$; pooled $\hat p = 580/4500 = 0.12889$.

**Step 2 - standard error of the difference.** $\text{SE} = \sqrt{\hat p(1-\hat p)\left(\frac{1}{2000}+\frac{1}{2500}\right)} = \sqrt{0.12889\times0.87111\times(0.0005+0.0004)}$. The product $0.12889\times0.87111 = 0.11228$, and the bracket $=0.0009$, so $\text{SE} = \sqrt{0.00010105} = 0.010052$.

**Step 3 - the z-statistic.** Observed difference $= 0.136-0.120 = 0.016$. Then $z = 0.016/0.010052 = 1.592$. A two-sided p-value for $z=1.592$ is about $0.111$.

**Step 4 - interpret in the sequential-testing context.** At a fixed one-shot sample this result is *not* significant at the conventional 0.05 level - you cannot reject "no difference." But this is exactly where the chapter's sequential-testing warning bites: if you had been peeking at the dashboard daily for a month, the most extreme day's z-score could easily have crossed 2 purely by multiple looks, and you would have falsely "found" an effect. The honest report is "inconclusive so far, and you must pre-register how many looks you'll take."

**Step 5 - connect to causal inference.** Even a clean significant result here only licenses $P(Y\mid \text{observed treatment})$ language; if users could self-select into the variant, you'd need the do-calculus or a propensity-score adjustment to claim $P(Y\mid \text{do(treatment)})$. Randomization is what makes "observed" and "intervened" coincide.

**The takeaway:** a single A/B test number is never the whole story - its z-score is only trustworthy if you've handled how many times you looked, and it's only causal if assignment was genuinely random.

## Exercises

### Exercise 1 - Compute a two-proportion z-test
Two variants: A has 500 sessions and 50 conversions; B has 500 sessions and 70 conversions. Compute the z-statistic for the difference in conversion rates and say whether it's significant at a two-sided 0.05 level (critical value ≈1.96).

<details><summary>Show solution</summary>

Rates: A: $50/500 = 0.10$; B: $70/500 = 0.14$; observed difference $= 0.04$.
Pooled proportion: $(50+70)/(500+500) = 120/1000 = 0.12$.
SE $= \sqrt{0.12\times0.88\times(1/500+1/500)} = \sqrt{0.1056\times0.004} = \sqrt{0.0004224} = 0.020553$.
$z = 0.04/0.020553 = 1.946$.

Since $1.946 < 1.96$, the result just misses the two-sided 0.05 threshold - inconclusive. A change this close to the line is exactly the situation where sequential peeking or a slightly different sample would flip your conclusion, so you'd pre-commit to a larger fixed sample before deciding.

</details>

### Exercise 2 - Diagnose correlation-vs-causation
A company observes that customers who enable a "power user" setting spend 30% more. They conclude the setting causes higher spend and decide to push it to everyone. What causal fallacy is this, and how would you design a study to fix it?

<details><summary>Show solution</summary>

This is confounding by self-selection: users who bother to enable a power setting are plausibly already more engaged/wealthier than average, so the setting may be a *marker* of high spenders, not a *cause* of spending. The naive comparison estimates $P(\text{spend}\mid \text{setting}=1)$ but they want $P(\text{spend}\mid \text{do(setting}=1))$.

Fix with a randomized experiment: randomly assign some users to see and be nudged toward the setting (treatment) and others to not (control), then compare spend. Randomization breaks the link between the setting and pre-existing engagement, so any post-assignment difference is causal. If a full experiment is impossible, use an instrumental variable or propensity-score matching on observed confounders - but only randomization (or a credible natural experiment) removes *unobserved* confounding.

</details>

### Exercise 3 - Order a learning path
Given a single thread to deepen - say Chapter 24's MCMC - list two concrete next topics and explain, for each, which chapter's tool it extends.

<details><summary>Show solution</summary>

Two natural continuations:

1. **Probabilistic programming (Stan, PyMC, NumPyro).** This extends Chapters 17 and 24 directly: it automates the Metropolis-Hastings/HMC sampling machinery so you can declare a model and let the tool build the Markov chain. Its Hamiltonian Monte Carlo improves on the naive random-walk proposals by using *gradients* - concretely, Chapter 0's derivative-as-slope idea - to propose smarter moves, which is why it's far more efficient in high dimensions.

2. **Variational inference.** This extends Chapters 17, 23, and 24 in a different direction: rather than sampling the posterior, it turns the inference problem into an optimization problem (Chapter 10's MLE-style fitting), trading exactness for speed. It is the engine behind Bayesian neural networks and the "ELBO" objective in VAEs (Chapter 23).

Both are legitimate forks of the same Chapter 24 trunk - one making sampling faster, the other replacing sampling with a sibling optimization algorithm.

</details>

## Why This Matters - In AI and In Everyday Life

**In AI / machine learning:**

- **Online experimentation drives product AI decisions**: virtually every feature shipped by a large tech platform - from a ranking tweak to a new model endpoint - is validated by the A/B testing machinery this chapter points toward, and the sequential-testing subtleties are everyday practitioner concerns.
- **Causal reasoning is becoming a core model capability**: frontier LLMs are increasingly expected to answer "what would happen if..." counterfactual questions and to respect intervention vs. observation, which is the do-calculus distinction this chapter introduces; causal inference is one of the most active research bridges between statistics and AI.

**In everyday, non-AI life:**

- **Reading a news headline about a "link" between habits and outcomes**: knowing the correlation-vs-causation distinction this chapter flags is what lets you tell a genuine finding from a confounded observational study in the wild.
- **Choosing what data to collect before a decision**: whether it's a business pilot, a clinical trial, or a personal budget experiment, designing the measurement (sample size, randomization, pre-registration) before looking at results is the everyday form of experimental design.

## A closing thought on how to keep learning this

The single habit this book has tried hardest to instill, across every chapter: when you meet a new formula, ask the three questions that structured this entire book - **what problem was this actually trying to solve, what did the person who invented it try first that didn't quite work, and what does this look like if I simulate it myself and watch the numbers move?** Formulas that feel like arbitrary incantations almost always have a genuinely traceable, often surprisingly human and improvisational origin story once you go looking for it - and a few hundred lines of Python can turn almost any abstract claim in statistics into something you've watched happen with your own eyes, which is a far more durable form of understanding than memorizing the formula alone ever provides. You now have the full vocabulary and the habit of mind to go find those origin stories and run those simulations yourself, for whatever comes next.
