# Chapter 10: Maximum Likelihood Estimation

## The idea this chapter builds toward

Instead of guessing estimator formulas ad hoc ("let's just average the data" — why that formula and not another?), **Maximum Likelihood Estimation (MLE)** gives you a single, general *principle* for constructing the "best" estimator for any distribution's parameters: **pick the parameter values that make the data you actually observed as likely as possible.** This isn't just an abstract idea — it is, almost without exaggeration, the mathematical skeleton underneath how every standard machine learning loss function (mean squared error, cross-entropy, logistic loss) is derived. Understanding MLE is understanding *why* those specific loss functions were chosen, not just what they compute.

## The likelihood function — flipping probability around

Recall a distribution's PDF/PMF, $f(x \mid \theta)$, tells you the probability (density) of observing data value $x$, *given* a known parameter $\theta$ (like the mean $\mu$ or rate $\lambda$). The **likelihood function** flips the perspective: you now have *fixed, observed data*, and you treat $\theta$ as the unknown you're searching over:

$$L(\theta \mid x) = f(x \mid \theta)$$

Same formula, same numbers — but a different question. The PDF asks "given this $\theta$, how likely is this $x$?" The likelihood asks "given this $x$ (which already happened), how plausible is each candidate $\theta$?" This is exactly the $P(E\mid H)$-vs-$P(H\mid E)$ asymmetry from Chapter 4's Bayes' Theorem — likelihood is literally the $P(E \mid H)$ piece of that formula, just now applied with continuous parameters instead of two Bayes alternatives.

For multiple independent observations $x_1, \ldots, x_n$ (independence lets us multiply, from Chapter 3):

$$L(\theta \mid x_1,\ldots,x_n) = \prod_{i=1}^n f(x_i \mid \theta)$$

**MLE's principle**: find the $\hat\theta$ that maximizes $L(\theta)$ — the parameter value under which the data you actually saw was the *most probable* outcome.

## Why we take the log (and why this connects back to Chapter 0)

Multiplying many probabilities (each less than 1) together produces an extremely tiny number that underflows on a computer, and is hard to differentiate (take the derivative of, to find the maximum) by hand. Since $\log$ is a strictly increasing function (Chapter 0), the $\theta$ that maximizes $L(\theta)$ is the *exact same* $\theta$ that maximizes $\log L(\theta)$. So we maximize the **log-likelihood** instead:

$$\log L(\theta) = \sum_{i=1}^n \log f(x_i \mid \theta)$$

Using $\log(a\times b) = \log a + \log b$ from Chapter 0, the product has become a sum — numerically stable, and each term's derivative is manageable. This is why you will see "log-likelihood" instead of plain "likelihood" absolutely everywhere in statistics and ML — it is never a different concept, just the numerically sane way to work with the same quantity.

## Worked example: deriving that the sample mean IS the MLE for a Normal distribution's mean

This is one of the most satisfying derivations in this book, because it proves the formula you've been using since Chapter 2 wasn't arbitrary — it falls directly out of the MLE principle.

Suppose $x_1, \ldots, x_n$ are drawn independently from $N(\mu, \sigma^2)$ with $\sigma$ known, and we want the MLE for $\mu$. The log-likelihood (dropping the terms not involving $\mu$, since they don't affect which $\mu$ maximizes it) is:

$$\log L(\mu) = -\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2 + \text{constant}$$

To find the maximizing $\mu$, take the derivative with respect to $\mu$ and set it to zero (standard calculus move: at a maximum, the slope is flat):

$$\frac{d}{d\mu}\log L(\mu) = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$$

$$\implies \sum_{i=1}^n (x_i - \mu) = 0 \implies \sum x_i = n\mu \implies \mu = \frac{1}{n}\sum x_i = \bar x$$

**The MLE for the Normal mean is exactly the sample mean.** This isn't a coincidence or a separate justification — "take the average" was secretly "find the most likely parameter" the entire time, for any data that's roughly Normally distributed (and by the CLT from Chapter 8, that's an extremely common situation).

## The deeper payoff: MLE is why your loss functions look the way they do

### Mean Squared Error = Gaussian MLE

Look again at the expression we maximized above: $-\frac{1}{2\sigma^2}\sum(x_i-\mu)^2$. Maximizing this is the same as *minimizing* $\sum (x_i - \mu)^2$ — the **sum of squared errors**. This means: **assuming your prediction errors are Normally distributed, minimizing mean squared error (MSE) in a regression model is mathematically identical to maximizing likelihood.** This is *why* MSE was chosen as a standard regression loss — not an arbitrary convenient formula, but the MLE-correct choice under a Gaussian error assumption. (If your errors are *not* roughly Normal — e.g., heavy-tailed with occasional huge outlier errors — MSE is no longer the MLE-justified choice, and a different loss, like mean absolute error, may be more appropriate — a very practical modeling decision this derivation directly informs.)

### Cross-entropy loss = Bernoulli/Categorical MLE

For a binary classifier predicting probability $\hat p$ that a label is 1, assuming each label is a Bernoulli trial (Chapter 5) with that predicted probability, the likelihood of observing the true label $y \in \{0,1\}$ is $\hat p^{\,y}(1-\hat p)^{1-y}$. Taking the log and negating (convention: we minimize *negative* log-likelihood rather than maximize log-likelihood — same optimum, just flipped sign) gives exactly:

$$\text{loss} = -\big[y\log\hat p + (1-y)\log(1-\hat p)\big]$$

This is **binary cross-entropy loss** — the single most common loss function for classification in all of deep learning. It was not invented separately from statistics; it is the MLE-correct loss for a Bernoulli-distributed label, derived by the exact same recipe as MSE above, just with a different assumed distribution for the data. This is the single biggest "aha" this book offers: **choosing a loss function is secretly choosing a probability distribution for your data/errors, and the loss function is just its negative log-likelihood.** We'll return to this precise connection again in Chapter 25.

## The backstory — Fisher's single unifying idea (1912, sharpened 1922)

As a 22-year-old undergraduate in 1912, Fisher wrote a short paper proposing what he then called the "method of the most probable value" — the same core idea you learned above, though he hadn't yet worked out its full theoretical justification. A decade later, in his landmark 1922 paper "On the Mathematical Foundations of Theoretical Statistics," he renamed it **maximum likelihood** and did something genuinely remarkable: he showed that this *single, general recipe* — "pick the parameter that makes the observed data most probable" — reproduces, as special cases, a whole scattering of previously separate, individually-derived estimation formulas that statisticians had cobbled together over the preceding century (including Gauss's least-squares approach from Chapter 6, and the sample proportion for Binomial data). Before Fisher, each distribution effectively needed its own bespoke, cleverly-reasoned estimator, discovered somewhat independently. After Fisher, you had one universal procedure that mechanically *generates* the right estimator for any distribution you can write a likelihood for — an enormous unification, philosophically similar to how Newton's laws of motion unified previously separate rules for falling bodies, planetary orbits, and pendulums into one framework. This is exactly why MLE, not some competing ad hoc principle, became the default estimation tool taught in every statistics course since — and exactly why, a century later, it turns out to be the same object underlying loss-function design in deep learning, a field Fisher could not have anticipated but whose mathematical bones he effectively built.

## MLE's known weaknesses (so you don't over-trust it)

MLE can overfit with small samples (it has no built-in preference for "simple" explanations — it will happily fit noise if that noise makes the observed data maximally likely) and it can behave badly for some distributions with small sample sizes (biased in finite samples, even though it's usually *asymptotically* unbiased and consistent as $n\to\infty$). The fix for the overfitting weakness — adding a preference for simpler/more plausible parameters — is exactly what Bayesian statistics (Chapter 17) and regularization (Chapter 16) formalize, by combining the likelihood with a *prior* belief (Chapter 4's Bayes' Theorem, reappearing here in a new costume).

## More real-world examples of MLE

- **Logistic regression** (binary classification): fits coefficients by maximizing the Bernoulli log-likelihood described above — every time you call `.fit()` on a logistic regression model, you're running MLE.
- **Language models**: training a next-word-prediction model by minimizing cross-entropy loss over a huge text corpus is, exactly, maximizing the likelihood of the observed training text under the model's predicted probability distribution over words — "the model that makes the real training data as unsurprising as possible" is a plain-English restatement of MLE.
- **Fitting a distribution to insurance claims data**: an actuary fitting, say, a Gamma distribution to historical claim sizes to estimate future risk uses MLE to find the Gamma parameters that make the observed claims history most probable.
- **A/B test conversion rate estimation**: the "obvious" estimator (observed conversions / total visitors) for a Binomial's $p$ is not a coincidence — it is exactly the MLE for Binomial data, derivable by the same log-likelihood-derivative-equals-zero recipe used above for the Normal mean.

## Figures

![The log-likelihood of Normal data as a function of candidate mu values. The peak sits exactly at the sample mean -- proof that the MLE for a Normal mean IS the sample mean.](../images/ch10_maximum_likelihood_estimation_fig1.png)

*The log-likelihood of Normal data as a function of candidate mu values. The peak sits exactly at the sample mean -- proof that the MLE for a Normal mean IS the sample mean.*

## Run it yourself

`book/code/ch10_maximum_likelihood_estimation.py` generates data from a known Normal distribution, plots the log-likelihood as a function of candidate $\mu$ values (so you can see the "hill" being climbed and confirm its peak sits exactly at $\bar x$), and then demonstrates the MSE-as-Gaussian-MLE connection concretely by fitting a simple linear regression two ways — once via `scipy.optimize` maximizing Gaussian log-likelihood directly, once via plain least-squares — and showing they land on identical parameters.

```bash
python book/code/ch10_maximum_likelihood_estimation.py
```

## Why This Matters — In AI and In Everyday Life

**In AI / machine learning:**

- **Every standard neural network loss function**: MSE, binary cross-entropy, and categorical cross-entropy are all literally negative log-likelihoods under a specific distributional assumption -- this chapter is the direct derivation of how deep learning training objectives were chosen.
- **Logistic regression and most classical ML model fitting**: calling `.fit()` on a logistic regression or many other classical models is, under the hood, running exactly the MLE recipe from this chapter.

**In everyday, non-AI life:**

- **Estimating a true defect rate from inspected samples**: if you inspect 50 items and find 3 defective, the "obvious" estimate of 6% defect rate is literally the MLE for a Binomial parameter, derivable by this chapter's recipe.
- **Fitting any theoretical curve to real measurements**: an actuary fitting a claim-size distribution to historical insurance claims, or a physicist fitting a decay curve to measurements, is running MLE.

## Check your understanding

- Why does maximizing $\log L(\theta)$ give the same answer as maximizing $L(\theta)$ directly?
- Walk through, in your own words, why setting the derivative of the log-likelihood to zero and solving gives $\mu = \bar x$ for Normal data.
- If someone tells you "we use MSE because it's the standard choice," what deeper statistical assumption are they implicitly making about their errors, according to this chapter?

## What's next

Chapter 11 uses the CLT (Chapter 8) plus the sample mean's known standard error (Chapter 9) to build **confidence intervals** — a precise way to express "how wrong might this estimate be" rather than reporting a single bare number.
