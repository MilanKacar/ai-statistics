# Chapter 0: How to Use This Book, and the Math Notation You'll See

## Why statistics, if you want to understand AI?

Every machine learning model is, underneath, a statistical statement: "given this data, what's the most plausible pattern?" Loss functions are disguised probability statements. Regularization is a disguised prior belief. Overfitting is a disguised statistics problem (confusing noise for signal). "Confidence" of a classifier is a disguised probability. If you understand statistics deeply, half of deep learning stops looking like magic and starts looking like consequences of a few simple ideas applied over and over.

> **Intuition, in one breath:** Machine learning is applied statistics wearing a different outfit - the same handful of ideas, renamed, power almost everything a model does.

This book's promise to you: by the end, you will not just know the *names* of these ideas (mean, variance, p-value, entropy, Bayes' theorem, maximum likelihood) - you'll know *why* each one is defined the way it is, what problem it's solving, and what it looks like in code.

## You said your math is rusty - here's the only notation you need right now

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

$f(x)$ is just a machine: you put a number in, a number comes out, by some rule. $f(x) = x^2$ means "square whatever you give me." Nothing deeper than that - in Python, it's literally `def f(x): return x**2`.

### Exponents and logarithms - the two you must be fluent in

**Exponent**: $a^b$ means multiply $a$ by itself $b$ times ($2^3 = 8$). In stats we constantly use $e^x$ where $e \approx 2.71828$ - a special constant that shows up because of how continuous growth/decay works (same constant as compound interest taken to the limit).

**Logarithm**: $\log(x)$ asks "e to what power gives me x?" It's the *undo* button for exponentiation. Two properties you'll use constantly:

$$\log(a \times b) = \log(a) + \log(b)$$
$$\log(a^b) = b \times \log(a)$$

Why do we care? Because in statistics we constantly multiply many small probabilities together (e.g., the probability of seeing this *entire dataset* under some model = product of per-point probabilities), and multiplying hundreds of small numbers together underflows to zero on a computer and is hard to reason about by hand. Logs turn that product into a **sum**, which is numerically stable and easy to differentiate. You will see "log-likelihood" everywhere starting in Chapter 10 - now you know why we bother taking the log at all.

### Derivatives, in one paragraph, intuitively

A derivative answers: "if I nudge the input a tiny bit, how much does the output move, and in which direction?" That's it - it's a *slope*. If $f(x) = x^2$, its derivative is $f'(x) = 2x$, meaning: at $x=3$, nudging $x$ up slightly increases $f$ about $6\times$ as fast. In ML, "training" literally means: compute the derivative (slope) of the error with respect to each parameter, and nudge the parameter in the direction that makes the error smaller. We'll use derivatives sparingly in this book (mainly in Chapter 10 and Chapter 16), and every time, I'll explain exactly what's being nudged and why.

## Where this notation itself came from (so it stops feeling arbitrary)

Even notation has a discovery story, and knowing it removes some of the "why does math look like THIS" mystery. The capital Greek sigma $\sum$ for summation was introduced by Leonhard Euler in 1755 - he simply picked the Greek letter matching the first sound of "sum," the same naming logic a programmer uses when they write a variable called `sum_total` instead of `x7`. The $\int$ integral sign, which you won't need in this book but will see referenced, is literally an elongated "S" for "Summa" (Latin for sum) - Leibniz drew it that way in 1675 because an integral is, at heart, a sum of infinitely many infinitesimally thin slices, and he wanted the symbol itself to remind you of that fact every time you saw it. The equals sign "=" was invented by Robert Recorde in 1557, who explicitly justified it by saying no two things could be more equal than a pair of parallel lines - a deliberately intuitive visual pun, not an arbitrary squiggle. The lesson worth carrying forward: **every piece of math notation in this book was invented by a specific person trying to make a specific idea easier to write down and think about - if a symbol ever feels like decoration, you're missing the shorthand it was built to provide, and it's worth pausing to ask what work it's actually doing.**

## The two kinds of statistics - the map for this whole book

1. **Descriptive statistics** (Chapters 1-2): summarizing data you already have. "What does this pile of numbers look like?"
2. **Probability** (Chapters 3-8): the math of uncertainty itself, used as a *language* to describe randomness, before we ever touch real data.
3. **Inferential statistics** (Chapters 9-15): using a *sample* of data to make justified claims about a *population* you can't fully observe. This is the heart of "classical" statistics, and it's also exactly what training a model on a training set and hoping it works on new data *is*.
4. **The ML/AI layer** (Chapters 16-26): where these ideas get reused, renamed, and extended into the machinery that makes modern AI work - bias-variance becomes "overfitting," MLE becomes "loss functions," entropy becomes "cross-entropy loss," Bayesian priors become "regularization," and so on.

## A promise about rigor vs. intuition

I will always give you the intuitive story *first*, then the formula, then a worked-by-hand numeric example, then code. If at any point a formula appears before you've gotten the intuition, that's a bug - tell me and I'll fix the chapter. You are allowed to ask "wait, why is it defined like *that* and not some other way?" at every single step - that question is exactly how you get to 100% understanding rather than 80%.

## Worked example - by hand, larger

Suppose a model's training set has 5 examples, and we want the mean loss (the sample mean). The per-example losses are $0.2, 0.6, 0.4, 0.8, 1.0$. Add them: $0.2 + 0.6 + 0.4 + 0.8 + 1.0 = 3.0$. Divide by $n = 5$: $\bar{x} = \frac{3.0}{5} = 0.6$. Now suppose the same model, evaluated on a *different* random draw of 5 examples, gives losses $0.1, 0.3, 0.9, 1.1, 0.6$. Sum $= 3.0$ again? No - $0.1+0.3+0.9+1.1+0.6 = 3.0$. Coincidence aside, notice both averages happen to be $0.6$ here, but with different *spread*.

Now write the same five losses in the newly introduced `sum` notation. For the second batch, $\sum_{i=1}^{5} x_i = 0.1+0.3+0.9+1.1+0.6 = 3.0$, so $\bar{x} = \frac{1}{5}\sum_{i=1}^{5} x_i = \frac{3.0}{5} = 0.6$. If instead we had used the log-likelihood trick from the notation section, we would write the product of per-example likelihoods $0.9 \times 0.7 \times 0.5 \times 0.4 \times 0.6$. Multiply step by step: $0.9 \times 0.7 = 0.63$; $0.63 \times 0.5 = 0.315$; $0.315 \times 0.4 = 0.126$; $0.126 \times 0.6 = 0.0756$. Taking the log of a product: $\log(0.0756)$. Rather than a calculator, note $\log(0.0756) \approx -2.582$ (base $e$); the crucial point is that we could equally have added the five individual logs, $\log 0.9 + \log 0.7 + \log 0.5 + \log 0.4 + \log 0.6$, and gotten the identical total - exactly why logs turn a fragile product of many small numbers into a stable sum.

**The whole notation section, worked once on numbers you can check by hand, is just a `for` loop: sum means add, product means multiply, and log means undo the multiply so you can add instead.**

## Exercises

### Exercise 1 - Evaluate a summation and a product

Given $x_1 = 2, x_2 = 3, x_3 = 5$, compute $\sum_{i=1}^{3} x_i$ and $\prod_{i=1}^{3} x_i$ by hand.

<details><summary>Show solution</summary>

Sum: $2 + 3 + 5 = 10$. Product: $2 \times 3 \times 5 = 30$. In Python these are `sum([2,3,5])` and `np.prod([2,3,5])`. The point is that both operations run over the exact same list; only the combining operator changes.

</details>

### Exercise 2 - Why logs help with many probabilities

You multiply together 5 independent per-example probabilities, each around 0.5. Roughly what is the product, and why would you rather sum their logs?

<details><summary>Show solution</summary>

The product is roughly $0.5^5 = 1/32 \approx 0.031$. With 50 such values, $0.5^{50} \approx 8.9 \times 10^{-16}$, which is tiny but survivable; with 500 values it is about $10^{-151}$, which underflows to zero in floating point. Summing the logs, by contrast, gives $-500 \log 2 \approx -346.6$, a perfectly ordinary number. That is the entire reason log-likelihoods exist: the same ranking (which model fits better) is preserved, but the arithmetic stays stable.

</details>

### Exercise 3 - Mapping ML words back to statistics words

For each ML term, name the statistics concept it renames: (a) loss function, (b) regularization, (c) overfitting, (d) classifier confidence.

<details><summary>Show solution</summary>

(a) Loss function - a negative log-likelihood or expected-error quantity (statistics: cost/risk). (b) Regularization - a prior belief penalizing complexity (statistics: a prior or a penalty on the estimator). (c) Overfitting - confusing sample noise for population signal (statistics: estimation error / variance not generalizing). (d) Classifier confidence - a probability statement about the label (statistics: a conditional probability, $P(\text{label} \mid \text{input})$). None of these is new math; each is an old idea with a new label.

</details>

## What's next

Chapter 1 starts at the very beginning: what actually *is* data, and the single most important distinction in all of statistics (population vs. sample) - the one that, misunderstood, causes most bad conclusions in both classical statistics and in machine learning (train/test splits are the same idea wearing a different hat).
