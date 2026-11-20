# Chapter 17: Bayesian Statistics in Depth

## The idea this chapter builds toward

Every chapter since 9 has used the **frequentist** framework: a true parameter is a fixed (if unknown) constant, and probability statements describe the long-run behavior of a *procedure* across hypothetical repeated samples (this is exactly why Chapter 11's confidence interval and Chapter 12's p-value both have those slightly awkward, indirect interpretations). This chapter presents the alternative, coherent framework hinted at since Chapter 4: treat the unknown parameter itself as a random variable with its own full probability distribution, representing your *degree of belief*, and update that belief with data using Bayes' Theorem directly. This isn't a totally separate subject — it's the same mathematical building blocks (Bayes' Theorem from Chapter 4, likelihood from Chapter 10) assembled into a different, internally consistent philosophy, with genuinely different practical answers to some questions.

## The frequentist/Bayesian divide, made concrete

Frequentist question: "If I repeated this experiment infinitely many times, what fraction of constructed intervals would contain the true $\mu$?" (A statement about the *procedure*.)

Bayesian question: "Given the data I actually observed, what is the probability distribution of $\mu$ itself?" (A direct statement about the *parameter*, treated as uncertain in the same sense a weather forecast treats tomorrow's rain as uncertain.)

Neither is "more correct" in some absolute sense — they're different, internally valid mathematical frameworks answering different questions, suited to different situations. You already met this fork in Chapter 4's backstory (Bayes and Laplace vs. the many mathematicians uneasy with assigning probabilities to fixed unknown facts) — this chapter is where that two-century-old disagreement gets fully resolved into precise, usable machinery.

## The Bayesian recipe, step by step

### 1. Choose a prior: $P(\theta)$

Before seeing any data, express your belief about the parameter $\theta$ as a full probability distribution. This might be based on previous studies, domain expertise, or deliberately chosen to be "weak"/uninformative if you want the data to speak almost entirely for itself.

### 2. Compute the likelihood: $P(\text{data}\mid\theta)$

Exactly Chapter 10's likelihood function — how probable is the data you actually observed, for each candidate value of $\theta$.

### 3. Combine via Bayes' Theorem to get the posterior: $P(\theta \mid \text{data})$

$$P(\theta \mid \text{data}) = \frac{P(\text{data}\mid\theta) \, P(\theta)}{P(\text{data})} \propto P(\text{data}\mid\theta)\,P(\theta)$$

Since the denominator $P(\text{data})$ doesn't depend on $\theta$ (it's just a normalizing constant, as in Chapter 4), it's common to write this as a proportionality: **posterior ∝ likelihood × prior**, the single sentence summary you first saw in Chapter 4, now applied to estimating an entire continuous parameter's distribution rather than choosing between two discrete hypotheses.

## Worked example — estimating a coin's bias, by hand, using a Beta prior

Suppose you want to estimate $p$, the true probability a coin lands heads, and you flip it 10 times, observing 7 heads. Being honest that you have mild prior belief the coin is probably close to fair (not a strong belief — you're open to being wrong), you choose a **Beta(2,2) prior** — a standard choice for probabilities (bounded between 0 and 1), whose shape is a gentle hump centered at 0.5 (chosen here specifically because it is mathematically *conjugate* to the Binomial likelihood — see the box below — making the update trivially easy to compute by hand, which is exactly why Beta priors are the standard textbook choice for this exact problem).

The Binomial likelihood for $k=7$ heads out of $n=10$ is proportional to $p^7(1-p)^3$ (dropping the $\binom{n}{k}$ constant, since it doesn't depend on $p$ and so doesn't affect the shape of the posterior). The Beta(2,2) prior density is proportional to $p^{2-1}(1-p)^{2-1} = p(1-p)$. Multiplying likelihood × prior:

$$\text{posterior} \propto p^7(1-p)^3 \times p(1-p) = p^{8}(1-p)^{4}$$

This is recognizably (up to a normalizing constant) a **Beta(9, 5)** distribution — using the general pattern Beta$(\alpha,\beta)$'s density $\propto p^{\alpha-1}(1-p)^{\beta-1}$, matching exponents $\alpha-1=8 \Rightarrow \alpha=9$ and $\beta-1=4\Rightarrow\beta=5$. The posterior mean of a Beta$(\alpha,\beta)$ is $\frac{\alpha}{\alpha+\beta}$, so our updated best estimate of $p$ is $\frac{9}{14}\approx 0.643$ — notice this sits *between* the prior's belief (0.5) and the raw data's suggestion ($7/10=0.7$), pulled toward 0.5 by the prior's gentle skepticism, exactly the "posterior as a weighted compromise between prior and data" intuition that the Bayesian framework formalizes everywhere.

### Why conjugate priors made Bayesian statistics computationally tractable for 200 years

Notice the genuinely convenient coincidence above: a Beta prior, combined with a Binomial likelihood, produces *another* Beta distribution as the posterior — this is called a **conjugate prior** relationship, and it's why Beta/Binomial (and similarly, Normal/Normal, and Gamma/Poisson) pairings are the textbook workhorses of Bayesian statistics: the algebra stays in closed form, with no numerical integration required, across every stage of updating. For two centuries, this computational convenience wasn't a minor nicety — it was the *entire reason* Bayesian methods remained practically usable at all, because for almost any other prior/likelihood combination, computing the posterior exactly requires integrating over all possible $\theta$ values (recall Chapter 6: continuous probabilities require integrals), which has no closed-form solution for most realistic models with many parameters — a genuinely hard computational problem.

## Why Bayesian methods were a historical dead end for so long, and what revived them

Outside of a few conjugate-prior special cases, Bayesian statistics was largely impractical for complex, real-world, multi-parameter problems until computers became powerful enough (starting seriously in the late 1980s-1990s) to run **Markov Chain Monte Carlo (MCMC)** methods — a family of simulation techniques that *approximate* the posterior distribution numerically rather than requiring a closed-form solution, by cleverly sampling from it even when its exact formula is intractable (full treatment in Chapter 24, once you have the Markov chain vocabulary needed to understand how it works). This computational unlock is a large part of why Bayesian methods, after roughly two centuries on the philosophical sidelines of mainstream (largely Fisher/Neyman-dominated frequentist) 20th-century statistics, surged back into widespread practical use from the 1990s onward — and it's also directly why Bayesian deep learning, a genuinely modern and still-developing subfield, needs serious computational machinery (variational inference, which approximates MCMC's job more cheaply) to be tractable at the scale of neural networks with millions of parameters.

## Credible intervals vs. confidence intervals — finally resolving Chapter 11's discomfort

A **95% credible interval** is the Bayesian posterior distribution's direct equivalent of a confidence interval, but with the straightforward interpretation Chapter 11 explicitly denied you: "given the data I observed, there is a 95% probability the true $\theta$ lies in this specific interval" — a perfectly legitimate statement *within the Bayesian framework*, precisely because $\theta$ is treated as a random variable with a genuine posterior probability distribution, not a fixed constant. If you found Chapter 11's frequentist interpretation unsatisfying, that discomfort reflected a real, valid intuition — you were just reaching for a Bayesian-flavored statement using frequentist machinery that isn't built to support it. With real data and weak/uninformative priors, credible intervals and confidence intervals often end up numerically very close — the disagreement is almost entirely philosophical/interpretational, not usually about the final numbers, which is worth knowing so you don't feel you must pick a "side" for every analysis.

## How regularization is secretly a Bayesian prior (promised in Chapter 16, now fully delivered)

Chapter 16 promised this connection: Ridge regression's $\lambda\sum\beta_j^2$ penalty is mathematically identical to performing MLE after assuming each $\beta_j$ has an independent $N(0,\tau^2)$ prior (for some $\tau$ related to $\lambda$) — maximizing the resulting posterior (not just the plain likelihood) is called **Maximum A Posteriori (MAP)** estimation, and it reproduces Ridge regression's loss function exactly. Lasso's $\lambda\sum|\beta_j|$ penalty similarly corresponds to a Laplace-distributed prior (a distribution with a sharper peak at zero than Normal, which is exactly why it pushes coefficients all the way to zero rather than just shrinking them). This means: **every time you choose a regularization strength, you are implicitly choosing a prior belief about how large your model's parameters should plausibly be** — regularization was Bayesian reasoning all along, just without the vocabulary to say so explicitly.

## Figures

![The posterior distribution for a coin's true bias p, sharpening and shifting toward the true value (black dashed line) as more coin flips accumulate -- the prior's influence visibly fades with more data.](../images/ch17_bayesian_statistics_fig1.png)

*The posterior distribution for a coin's true bias p, sharpening and shifting toward the true value (black dashed line) as more coin flips accumulate -- the prior's influence visibly fades with more data.*

## Run it yourself

`book/code/ch17_bayesian_statistics.py` reproduces the coin-bias Beta-Binomial example exactly as derived above, visualizes how the posterior shifts and sharpens as you feed it more and more coin flips (watch the prior's influence fade as data accumulates — a general Bayesian phenomenon worth seeing directly), and concretely demonstrates the Ridge-regression-as-MAP-estimation connection by fitting the same regression two ways — once via `sklearn`'s Ridge, once by directly maximizing a Gaussian-likelihood-times-Gaussian-prior posterior — and showing they match.

```bash
python book/code/ch17_bayesian_statistics.py
```

## Why This Matters — In AI and In Everyday Life

**In AI / machine learning:**

- **Bayesian hyperparameter optimization**: tools used throughout ML research treat each hyperparameter setting's performance as uncertain and use Bayesian updating to decide what to try next, directly using this chapter's machinery.
- **Cold-start recommendation systems**: when a new user or item has almost no data, a sensible prior belief (informed by similar users/items) combined with the little data available is exactly Bayesian updating in production.

**In everyday, non-AI life:**

- **Doctors updating a diagnosis as test results come in**: each new test result updates the posterior belief about what's wrong, which then becomes the prior for interpreting the next test -- real diagnostic reasoning, not just theory.
- **Weather forecasts updating through the day**: a morning forecast is a prior; afternoon radar data updates it to a posterior, which becomes the new prior for the evening forecast -- sequential Bayesian updating in daily life.

## Check your understanding

- In your own words, contrast what a frequentist confidence interval and a Bayesian credible interval are each actually claiming.
- Why does the posterior mean $\frac{9}{14}$ in the worked example sit between the prior's belief (0.5) and the raw observed proportion (0.7), and what happens to this balance as you collect much more data (say 1000 flips, 700 heads)?
- Explain, in your own words, why Ridge regression's L2 penalty corresponds to a Normal prior on the coefficients, using the posterior ∝ likelihood × prior logic.

## What's next

Chapter 18 steps back toward a more computational, less distributionally-assumption-heavy toolkit: **resampling methods** (the bootstrap and permutation tests), which let you estimate uncertainty and test hypotheses directly from your data via repeated simulation, often needed exactly when the clean-formula assumptions behind Chapters 11-13 don't quite hold.
