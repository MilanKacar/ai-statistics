# Chapter 0: How to Use This Book, and the Math Notation You'll See

## Why statistics, if you want to understand AI?

Every machine learning model is, underneath, a statistical statement: "given this data, what's the most plausible pattern?" Loss functions are disguised probability statements. Regularization is a disguised prior belief. Overfitting is a disguised statistics problem (confusing noise for signal). "Confidence" of a classifier is a disguised probability. If you understand statistics deeply, half of deep learning stops looking like magic and starts looking like consequences of a few simple ideas applied over and over.

This book's promise to you: by the end, you will not just know the *names* of these ideas (mean, variance, p-value, entropy, Bayes' theorem, maximum likelihood) — you'll know *why* each one is defined the way it is, what problem it's solving, and what it looks like in code.

## You said your math is rusty — here's the only notation you need right now

You don't need to memorize this. Just skim it once, and come back here whenever you hit a symbol you forgot.

### Summation (Σ)

$\sum_{i=1}^{n} x_i$ just means "add up a list of numbers." If you have data points $x_1, x_2, x_3 = 2, 5, 9$, then:

$$\sum_{i=1}^{3} x_i = 2 + 5 + 9 = 16$$

That's it. It's a `for` loop that accumulates a sum. In Python: `sum(x)` or `np.sum(x)`.

### Product (Π)

Same idea but multiplying instead of adding: $\prod_{i=1}^{n} x_i = x_1 \times x_2 \times \cdots \times x_n$. In Python: `np.prod(x)`.

### Set notation basics

- $x \in A$ means "x is an element of set A" (a collection of things).
- $P(A)$ means "the probability of event A."
- $|A|$ means "the number of elements in A" (if A is a finite set).

### Functions

$f(x)$ is just a machine: you put a number in, a number comes out, by some rule. $f(x) = x^2$ means "square whatever you give me." Nothing deeper than that — in Python, it's literally `def f(x): return x**2`.

### Exponents and logarithms — the two you must be fluent in

**Exponent**: $a^b$ means multiply $a$ by itself $b$ times ($2^3 = 8$). In stats we constantly use $e^x$ where $e \approx 2.71828$ — a special constant that shows up because of how continuous growth/decay works (same constant as compound interest taken to the limit).

**Logarithm**: $\log(x)$ asks "e to what power gives me x?" It's the *undo* button for exponentiation. Two properties you'll use constantly:

$$\log(a \times b) = \log(a) + \log(b)$$
$$\log(a^b) = b \times \log(a)$$

Why do we care? Because in statistics we constantly multiply many small probabilities together (e.g., the probability of seeing this *entire dataset* under some model = product of per-point probabilities), and multiplying hundreds of small numbers together underflows to zero on a computer and is hard to reason about by hand. Logs turn that product into a **sum**, which is numerically stable and easy to differentiate. You will see "log-likelihood" everywhere starting in Chapter 10 — now you know why we bother taking the log at all.

### Derivatives, in one paragraph, intuitively

A derivative answers: "if I nudge the input a tiny bit, how much does the output move, and in which direction?" That's it — it's a *slope*. If $f(x) = x^2$, its derivative is $f'(x) = 2x$, meaning: at $x=3$, nudging $x$ up slightly increases $f$ about $6\times$ as fast. In ML, "training" literally means: compute the derivative (slope) of the error with respect to each parameter, and nudge the parameter in the direction that makes the error smaller. We'll use derivatives sparingly in this book (mainly in Chapter 10 and Chapter 16), and every time, I'll explain exactly what's being nudged and why.

## The two kinds of statistics — the map for this whole book

1. **Descriptive statistics** (Chapters 1–2): summarizing data you already have. "What does this pile of numbers look like?"
2. **Probability** (Chapters 3–8): the math of uncertainty itself, used as a *language* to describe randomness, before we ever touch real data.
3. **Inferential statistics** (Chapters 9–15): using a *sample* of data to make justified claims about a *population* you can't fully observe. This is the heart of "classical" statistics, and it's also exactly what training a model on a training set and hoping it works on new data *is*.
4. **The ML/AI layer** (Chapters 16–26): where these ideas get reused, renamed, and extended into the machinery that makes modern AI work — bias-variance becomes "overfitting," MLE becomes "loss functions," entropy becomes "cross-entropy loss," Bayesian priors become "regularization," and so on.

## A promise about rigor vs. intuition

I will always give you the intuitive story *first*, then the formula, then a worked-by-hand numeric example, then code. If at any point a formula appears before you've gotten the intuition, that's a bug — tell me and I'll fix the chapter. You are allowed to ask "wait, why is it defined like *that* and not some other way?" at every single step — that question is exactly how you get to 100% understanding rather than 80%.

## What's next

Chapter 1 starts at the very beginning: what actually *is* data, and the single most important distinction in all of statistics (population vs. sample) — the one that, misunderstood, causes most bad conclusions in both classical statistics and in machine learning (train/test splits are the same idea wearing a different hat).
