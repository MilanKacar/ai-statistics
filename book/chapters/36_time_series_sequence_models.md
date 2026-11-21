# Chapter 36: Time Series and Modern Sequence Models - Statistics Over Time

## The idea this chapter builds toward

Almost everything in this book quietly assumed that your data points were **independent** - Chapter 3's independence assumption is the load-bearing wall underneath sampling (Chapter 9), the Central Limit Theorem (Chapter 8), regression (Chapter 14), and nearly every test since. This chapter takes a sledgehammer to that wall and rebuilds it for the one setting where it most obviously fails: **data ordered in time**, where today depends on yesterday. One idea carries the whole chapter: if the past is predictive, model the dependence directly - and the simplest such model, an **autoregression**, turns out to be just Chapter 14's regression pointed at its own past, Chapter 7's correlation of a series with itself, and Chapter 24's Markov chain wearing continuous clothes.

> **Intuition, in one breath:** yesterday tells you something about today, so stop pretending your data points are strangers - regress each value on its own recent history, and modern sequence models are that same idea scaled up until it can read a sentence.

## Why i.i.d. breaks, and what replaces it

Chapter 3 introduced the **i.i.d.** assumption - independent and identically distributed - as the costly but convenient fiction that makes almost all of classical inference work. Real sequential data breaks it constantly: a stock price today is related to yesterday's, a temperature reading is related to the one an hour ago, the next word in a sentence depends on the words before it. The "identical" half fails too, because a series with a trend is not drawing from the same distribution at the start as at the end.

The replacement is to model the **joint** distribution of the whole sequence, $P(x_1, x_2, \dots, x_T)$, rather than a product of identical marginals. Chapter 34 showed the general tool: the chain rule,

$$P(x_1,\dots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_1,\dots,x_{t-1}),$$

**Reading it piece by piece:**
- **$P(x_1,\dots,x_T)$** - the joint probability of the whole sequence, the thing every time-series model is really trying to describe.
- **$\prod_{t=1}^{T}$** - a product across time steps, multiplying the chance of each value given everything before it.
- **$P(x_t \mid x_1,\dots,x_{t-1})$** - the conditional probability of today's value given the entire past up to that point, the history-dependence each model simplifies.

which is always true but useless if each conditional depends on the entire history. The entire game of time-series modeling is choosing a **tractable, useful form for that history dependence** - and that choice is where every model in this chapter differs.

## Autocorrelation - a series correlated with its own past

Chapter 7 defined correlation between two variables. **Autocorrelation** is the same idea applied to one series against a shifted copy of itself. Fix a **lag** $h$ and correlate $x_t$ with $x_{t-h}$:

$$\rho_h = \text{Corr}(x_t, x_{t-h}) = \frac{\text{Cov}(x_t, x_{t-h})}{\sigma_{x_t}\sigma_{x_{t-h}}}.$$

**Reading it piece by piece:**
- **$\rho_h$** - the autocorrelation at lag $h$, a correlation that runs from -1 to 1 telling you how strongly a value moves with its own past.
- **$h$** - the lag, how many time steps back you are comparing.
- **$x_t$ and $x_{t-h}$** - today's value and the value $h$ steps earlier, the two series being correlated.
- **$\text{Cov}(x_t, x_{t-h})$** - how the two move together, positive when a high value now tends to follow a high value back then.
- **$\sigma_{x_t}\sigma_{x_{t-h}}$** - the product of the two standard deviations, which rescales the covariance into a unit-free number.

The **autocorrelation function (ACF)** is the plot of $\rho_h$ against $h$. A high $\rho_1$ means consecutive values move together; a slow decay of $\rho_h$ means long-range memory; a sharp cutoff after lag $q$ is the fingerprint of a moving-average model (below). This is the single most important diagnostic in the field - it is what tells you whether your series has memory worth modeling at all, and it is exactly Chapter 7's correlation computation run down a column of your data against the same column shifted by one.

## Stationarity - why the rules have to stop changing

A series is **stationary** if its statistical properties don't drift with time: constant mean, constant variance, and an autocorrelation that depends only on the lag $h$, not on the absolute time $t$. Stationarity matters because essentially every formula below assumes it - a stationary mean is a well-defined thing to forecast toward, whereas a trending series has a moving target.

The classic failure modes are a **trend** (mean drifts upward, like sales growing every year) and a **unit root** (shocks accumulate permanently rather than fading). The standard fix for both is **differencing**: replace $x_t$ with $\Delta x_t = x_t - x_{t-1}$, which removes a linear trend and is the "$I$" in ARIMA. This parallels Chapter 14's residual-plot discipline exactly: there, you checked homoscedasticity before trusting a regression; here, you check stationarity before trusting a time-series model.

## Autoregression - a regression on your own past

The cleanest stationary model is the **autoregression of order 1**, or **AR(1)**:

$$x_t = \phi x_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma^2) \text{ i.i.d.}$$

**Reading it piece by piece:**
- **$x_t$** - today's value of the series, the quantity being explained.
- **$\phi$** - the autoregressive coefficient, the slope of "today on yesterday"; a value with magnitude below 1 is what makes the series stationary.
- **$x_{t-1}$** - yesterday's value, the single predictor the model regresses on.
- **$\varepsilon_t$** - today's fresh innovation, the new surprise that no past value could have predicted.
- **$\varepsilon_t \sim N(0, \sigma^2)$ i.i.d.** - that innovation is drawn independently each step from a normal with mean zero and variance $\sigma^2$.

Read this through Chapter 14's lens and it is nothing new: it is a **simple linear regression of $x_t$ on $x_{t-1}$**, with intercept zero and slope $\phi$. The only twist is that the predictor is the series itself, one step back. The noise $\varepsilon_t$ is the fresh, unpredictable innovation at each step. For the series to be stationary you need $|\phi| < 1$ - if $|\phi| \ge 1$ each shock feeds back and either explodes or wanders forever (the unit-root case). An **AR($p$)** model extends this to regress on the last $p$ values at once, which is Chapter 15's *multiple* regression on lagged features.

The **moving-average**, or **MA($q$)**, model looks superficially similar but is conceptually different: it regresses $x_t$ not on past *values* but on past *shocks*,

$$x_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q}.$$

An MA model says "today is a baseline plus the last few surprises." **ARIMA($p,d,q$)** simply combines all three ingredients you now have: AR terms for memory in values, $d$ round(s) of differencing for trend/unit roots, and MA terms for memory in shocks. That acronym is not a new object to fear - it is the three pieces you just built, stacked.

## The AR(1) is a Markov chain - and its stationary variance

Notice what the AR(1) says: $x_t$ depends on the past *only through* $x_{t-1}$. That is precisely the **Markov property** from Chapter 24 - the future depends on the present, not the full history. An AR(1) is a Markov chain with a continuous state space. Everything Chapter 24 proved about stationary distributions now becomes a statement about the *distribution* of $x_t$: reachable long run, it settles into a fixed distribution regardless of where it started.

We can derive that distribution's mean and variance in three lines, using the rules from Chapter 7. Take expectations of $x_t = \phi x_{t-1} + \varepsilon_t$ and use $E[\varepsilon_t] = 0$ and the stationary fact $E[x_t] = E[x_{t-1}] = \mu$:

$$\mu = \phi\mu \quad\Rightarrow\quad (1-\phi)\mu = 0 \quad\Rightarrow\quad \mu = 0.$$

Take variances, remembering $\text{Var}(aX+b) = a^2\text{Var}(X)$ and that $\varepsilon_t$ is independent of $x_{t-1}$:

$$\text{Var}(x_t) = \phi^2 \text{Var}(x_{t-1}) + \sigma^2 \quad\Rightarrow\quad \gamma_0 = \phi^2\gamma_0 + \sigma^2 \quad\Rightarrow\quad \gamma_0 = \frac{\sigma^2}{1-\phi^2}.$$

**Reading it piece by piece:**
- **$\text{Var}(x_t)$** - the variance of today's value, what we are solving for.
- **$\phi^2 \text{Var}(x_{t-1})$** - yesterday's variance carried forward, scaled by the square of the coefficient because variance squares the multiplier.
- **$\sigma^2$** - the added variance of today's fresh innovation.
- **$\gamma_0$** - shorthand for the stationary variance, the common value the variance settles into at every step.
- **$1/(1-\phi^2)$** - the feedback multiplier; the closer $|\phi|$ is to 1, the more the noise is amplified, and the larger the long-run spread.

So a stationary AR(1) has stationary variance $\gamma_0 = \sigma^2/(1-\phi^2)$. As $|\phi| \to 1$ this blows up - a nearly unit-root series has a huge spread - and as $\phi \to 0$ it collapses to the noise variance, as it should. The autocorrelation at lag $h$ follows the clean geometric law $\rho_h = \phi^h$, so a positive $\phi$ gives a slowly decaying, all-positive ACF that is the signature of an AR(1).

Because $|\phi| < 1$, the series is **mean-reverting**: after any shock it decays back toward its mean by a factor $\phi$ each step. The **half-life** - how long until a deviation is half gone - comes from solving $\phi^h = 1/2$, giving $h = \ln(2)/\ln(1/\phi)$. This is the same mean-reversion that Chapter 7's "regression to the mean" described in cross-sectional data, now playing out as a function of time.

<div class="labcard">
<div class="labkicker">Live - simulate an AR(1) and watch its memory decay</div>
<div class="labcontrols">
  <div class="labctrl"><label>AR coefficient phi <b id="ch36_phi_label">0.80</b></label><input type="range" id="ch36_phi" min="-0.95" max="0.95" step="0.05" value="0.8"></div>
</div>
<div class="labrow">
  <button class="labbtn" id="ch36_run">Run simulation</button>
  <button class="labbtn secondary" id="ch36_reset">Reset</button>
</div>
<div class="labcanvas-wrap"><canvas id="ch36_canvas" width="820" height="240"></canvas></div>
<div class="labstat-grid">
  <div class="labstat"><div class="lbl">Empirical lag-1 autocorr</div><div class="val" id="ch36_acf1">-</div></div>
  <div class="labstat"><div class="lbl">Empirical lag-2 autocorr</div><div class="val" id="ch36_acf2">-</div></div>
  <div class="labstat"><div class="lbl">Stationary variance (theory)</div><div class="val" id="ch36_ssvar">-</div></div>
</div>
<p class="labnote">Drag phi toward the edges and rerun: near +/-1 the series drifts and smooths out, and the autocorrelations hover near phi and phi^2; near 0 it looks like pure noise with near-zero autocorrelation. Reset reseeds the random noise.</p>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var SL = window.StatsLab;
  var series = null;
  function autocorr(x, lag){
    var n = x.length;
    if(n <= lag+1) return 0;
    var m = SL.mean(x);
    var num = 0, den = 0;
    for(var t=0; t<n; t++){
      var d = x[t]-m; den += d*d;
      if(t>=lag) num += (x[t]-m)*(x[t-lag]-m);
    }
    return den > 1e-12 ? num/den : 0;
  }
  function generate(){
    var phi = parseFloat(document.getElementById('ch36_phi').value);
    var N = 1200;
    var x = new Array(N);
    x[0] = SL.randNormal(0,1);
    for(var t=1; t<N; t++){ x[t] = phi*x[t-1] + SL.randNormal(0,1); }
    series = x;
    document.getElementById('ch36_phi_label').textContent = phi.toFixed(2);
    var a1 = autocorr(x,1), a2 = autocorr(x,2);
    var ssvar = Math.abs(1-phi*phi) > 1e-6 ? 1/(1-phi*phi) : 0;
    document.getElementById('ch36_acf1').textContent = a1.toFixed(3);
    document.getElementById('ch36_acf2').textContent = a2.toFixed(3);
    document.getElementById('ch36_ssvar').textContent = ssvar.toFixed(3);
    draw(x);
  }
  function draw(x){
    var view = x.slice(Math.max(0, x.length-200));
    var lo = Math.min.apply(null, view), hi = Math.max.apply(null, view);
    if(!(hi > lo)) { lo -= 1; hi += 1; }
    var pad = (hi-lo)*0.1;
    var plot = SL.makePlot(document.getElementById('ch36_canvas'));
    plot.setDomain([0, view.length-1],[lo-pad, hi+pad]);
    plot.clear(); plot.axes('time (most recent 200 steps)','x_t');
    var pts = view.map(function(v,i){ return [i, v]; });
    plot.line(pts, SL.colors.accent, 2);
    plot.hline(SL.mean(view), SL.colors.accent4, true, 1);
  }
  document.getElementById('ch36_run').addEventListener('click', generate);
  document.getElementById('ch36_reset').addEventListener('click', generate);
  document.getElementById('ch36_phi').addEventListener('input', function(e){
    document.getElementById('ch36_phi_label').textContent = parseFloat(e.target.value).toFixed(2);
  });
  generate();
});
</script>

## State-space models and the Kalman filter

Time series also have **hidden state**: you observe a noisy measurement, but the underlying quantity you care about is unseen. Chapter 23's latent-variable theme meets Chapter 24's Markov chain here. A **state-space model** posits a hidden state $z_t$ evolving as a Markov chain, plus an observation $x_t$ that depends only on the current state:

$$z_t = \phi z_{t-1} + \eta_t \quad\text{(state transition)}, \qquad x_t = z_t + \nu_t \quad\text{(observation)}.$$

**Reading it piece by piece:**
- **$z_t$** - the hidden state at time $t$, the true quantity you care about but cannot see directly.
- **$\phi z_{t-1}$** - the hidden state carried forward from the previous step, here as a simple AR(1) transition.
- **$\eta_t$** - the state-transition noise, the random shock that jostles the hidden state each step.
- **$x_t$** - the observation, the noisy measurement you actually record.
- **$\nu_t$** - the observation noise, the measurement error separating the noisy reading from the clean hidden state.

The **Kalman filter** performs recursive Bayesian updating over $z_t$ - this is Chapter 4/17's Bayesian update, repeated once per time step instead of once ever. Each step has two phases: **predict** the hidden state forward using the transition, then **correct** it using the new observation, weighting prediction versus observation by their relative uncertainty (exactly Chapter 7's precision-weighted averaging). Because it only ever needs the current belief, it is fully recursive and linear-time - a Markov chain of beliefs. A Hidden Markov Model (Chapter 34's forward-backward machinery) is the discrete-state sibling of the same idea.

## Modern sequence models - the same statistics, scaled up

A **recurrent neural network (RNN)** is the nonlinear generalization of exactly this: a hidden vector $h_t = f(h_{t-1}, x_t)$ carries a summary of the past forward through time, learning the transition function instead of fixing it as $\phi z_{t-1}$. The AR(1)/state-space recursion is the RNN's linear special case.

The modern **state-space models** - **S4** (structured state-space sequence models) and **Mamba** - take the linear dynamical system above and make it *deep and input-dependent*: they stack many linear state-space layers (the $z_t = \phi z_{t-1} + \dots$ recurrence, sometimes letting $\phi$ vary with the input in Mamba), and their central technical trick is re-parameterizing the recurrence so it can be computed as a single convolution over the whole sequence. In other words, they are the statistics of linear dynamical systems - the AR(1), the Kalman filter's substrate - made trainable, deep, and fast. This is why they can process very long sequences: they are, at heart, the geometric decay of autocorrelation that this chapter started with.

**Transformers** (Chapter 29) sit beside these models rather than on top of them. They also model a sequence, but they abandon the recurrence entirely: **attention** lets every position look directly at every other position, typically with no notion of order except a positional encoding. That trades the AR(1)'s built-in Markov memory for an all-pairs, parallelizable operation - different statistics (attention weights, not decaying autocorrelation signatures), different strengths. The recurrence-versus-attention contrast *is* the contrast between "memory as a decaying Markov state" and "memory as a learned lookup over the entire past."

## Worked example - by hand, larger

Take a stationary AR(1) with $\phi = 0.8$ and innovation variance $\sigma^2 = 1$. We compute the stationary mean and variance, the autocorrelations at lags 1 and 2, and a one-step-ahead forecast.

**Stationary mean.** As derived above, $\mu = 0$. The series hovers around zero; there is nothing to forecast toward but its own mean.

**Stationary variance.** $\gamma_0 = \dfrac{\sigma^2}{1-\phi^2} = \dfrac{1}{1-0.8^2} = \dfrac{1}{1-0.64} = \dfrac{1}{0.36} \approx 2.7778$. So the stationary standard deviation is $\sqrt{2.7778} \approx 1.6667$. Notice this is much larger than the innovation spread of $1$: feedback magnifies the noise by the factor $1/(1-\phi^2)$.

**Autocorrelation at lag 1 and lag 2.** Using $\rho_h = \phi^h$: $\rho_1 = 0.8$ and $\rho_2 = 0.8^2 = 0.64$. The series is strongly, positively autocorrelated at short lags, decaying geometrically - the classic AR(1) ACF. The **half-life**: $h = \ln 2/\ln(1/0.8) = 0.6931/0.2231 \approx 3.11$ steps, so a shock is half gone in a bit over three steps.

**One-step-ahead forecast.** Suppose the present value is $x_t = 2.0$. The forecast of the next value is the conditional mean:

$$\hat x_{t+1} = \phi x_t = 0.8 \times 2.0 = 1.6.$$

Since $x_t = 2.0$ is above the stationary mean of $0$, the model forecasts a pull back down - that is mean-reversion in action.

**Forecast error variance.** The one-step error is $x_{t+1} - \hat x_{t+1} = \varepsilon_{t+1}$, so its variance is exactly $\text{Var}(\varepsilon_{t+1}) = \sigma^2 = 1$. The point forecast $1.6$ comes with a standard deviation of $1$. If we instead forecast $k$ steps ahead, the error accumulates as $\sigma^2(1-\phi^2 + \phi^4 + \dots)$ approaching the stationary variance $2.7778$ as $k \to \infty$ - the forecast's uncertainty grows until it is no better than knowing the long-run distribution.

**A 95% interval, to make it concrete.** Using the normal-approximation rule $\hat x \pm 1.96\sigma$ from Chapter 11, the one-step forecast interval is $1.6 \pm 1.96(1) = (-0.36,\ 3.56)$.

**Takeaway: a stationary AR(1) is entirely described by its mean (here 0), its stationary variance $\sigma^2/(1-\phi^2)$ (2.78), its geometric autocorrelation $\phi^h$, and an optimal forecast that is just the mean-reverting pull $\phi x_t$ - plus the beautiful fact that the one-step forecast error variance is simply the innovation variance $\sigma^2$ itself.**

## Check your understanding

- Why does a stationary AR(1) require $|\phi| < 1$, and what goes wrong statistically (not just numerically) when $|\phi| \ge 1$?
- Explain how the AR(1)'s stationary variance $\sigma^2/(1-\phi^2)$ and its geometric autocorrelation $\phi^h$ are two views of the same underlying memory.
- In what sense is the Kalman filter "Bayesian updating repeated over time," and what makes it recursive rather than needing the whole history?

## Exercises

### Exercise 1 - An AR(1) by hand

A stationary AR(1) has $\phi = 0.5$ and $\sigma^2 = 3$, with current value $x_t = 4$. Compute the stationary mean and variance, $\rho_1$ and $\rho_2$, the one-step-ahead forecast, and its forecast error variance.

<details><summary>Show solution</summary>

Stationary mean: $\mu = 0$ (same argument as the worked example - the intercept is zero).

Stationary variance: $\gamma_0 = \dfrac{\sigma^2}{1-\phi^2} = \dfrac{3}{1-0.25} = \dfrac{3}{0.75} = 4$, so the stationary standard deviation is $2$.

Autocorrelations: $\rho_1 = \phi = 0.5$ and $\rho_2 = \phi^2 = 0.25$.

One-step forecast: $\hat x_{t+1} = \phi x_t = 0.5 \times 4 = 2.0$. Since $4$ is far above the mean of $0$, the model predicts a strong pull back.

Forecast error variance: the one-step error is $\varepsilon_{t+1}$, so its variance is $\text{Var}(\varepsilon_{t+1}) = \sigma^2 = 3$, i.e. a forecast standard deviation of $\sqrt 3 \approx 1.732$.

Sanity check: the stationary variance ($4$) is larger than the innovation variance ($3$) because feedback magnifies noise, with the company of $\phi = 0.5$ adding a modest $1/(1-0.25) \approx 1.33$ multiplier.

</details>

### Exercise 2 - Why differencing removes a trend

Suppose a series grows linearly: $x_t = 5t + u_t$, where the $u_t$ are i.i.d. zero-mean noise. Show that the differenced series $\Delta x_t = x_t - x_{t-1}$ is stationary, and explain why the original series was not.

<details><summary>Show solution</summary>

Compute the difference directly:

$$\Delta x_t = x_t - x_{t-1} = (5t + u_t) - (5(t-1) + u_{t-1}) = 5 + u_t - u_{t-1}.$$

This depends on $t$ only through the noise terms $u_t - u_{t-1}$, whose distribution is the same at every $t$. So the mean is constant ($5$) and the variance is constant ($\text{Var}(u_t - u_{t-1}) = 2\sigma_u^2$), and the autocorrelation depends only on the lag - the differenced series is stationary.

The original series fails stationarity because its mean $E[x_t] = 5t$ grows without bound as $t$ increases: the statistical properties depend on absolute time, which is exactly what stationarity forbids. Differencing subtracts out the deterministic trend, leaving only the stationary noise structure. This is why the "$I$" (integration/differencing) step exists in ARIMA.

</details>

### Exercise 3 - Conceptual: AR(1) is a regression but not an ordinary one

Explain why fitting an AR(1) looks exactly like Chapter 14's simple linear regression of $x_t$ on $x_{t-1}$, yet the assumptions and inference are *not* the same as ordinary least squares - name the key assumption that breaks.

<details><summary>Show solution</summary>

Using lagged values as predictors, the AR(1) is geometrically identical to regressing $x_t$ on $x_{t-1}$ with no intercept (and, once you stack the lags, AR($p$) is exactly Chapter 15's multiple regression on $p$ lagged features). The slope estimate from least squares would even be the same algebra.

The assumption that breaks is **independence of the errors (no autocorrelation)**. In ordinary regression (Chapter 14), the residuals $\varepsilon_i$ are assumed independent across observations. In an AR(1) the observations are *generated by the model itself*, so the predictor $x_{t-1}$ is correlated with error terms from earlier in the series, and the residuals are not independent draws. This violates the standard OLS inference machinery - the estimates may be consistent, but the textbook standard errors, confidence intervals, and p-values are wrong. It is also Chapter 14's homoscedasticity-and-independence-of-errors checklist failing in a new way, which is precisely why time series needs its own estimation and testing tools rather than a naive `linregress(x_{t-1}, x_t)`.

</details>

## Why This Matters - In AI and In Everyday Life

**In AI / machine learning:**

- **Forecasting and control**: weather, demand, traffic, and financial prediction are time-series problems first, and the AR/ARIMA/state-space toolkit is still the interpretable baseline any deep forecaster must beat.
- **State-space sequence models (S4, Mamba)**: these are the linear dynamical systems of this chapter made deep and input-dependent, giving linear-time, very-long-context sequence modeling that competes with attention.
- **RNNs and the recurrence-versus-attention trade-off**: the hidden-state recurrence is the neural generalization of the AR(1), and understanding its Markov memory clarifies exactly what transformers (Chapter 29) give up and gain by replacing it with attention.
- **Reinforcement learning and diffusion chains**: RL trajectories and diffusion sampling (Chapters 30, 28) are both sequential processes whose dynamics are Markov chains over time - this chapter's vocabulary is the substrate.

**In everyday, non-AI life:**

- **Spikes that revert**: an unusually cold day, a viral post, or a bad sales quarter is usually followed by a pull back toward normal - an AR(1) with $|\phi| < 1$ watching a shock decay, not a permanent regime change.
- **Why a trend line can mislead**: projecting a growing metric forever (a series with a unit root) is the everyday version of the stationarity mistake - you must difference or detrend before believing the forecast.

## What's next

Where does the book go after "statistics over time"? Chapter 37 opens the final, most human part of the story: **aligning AI with human preferences**. Everything so far has trained a model against a fixed, given objective - a label, a likelihood, a reward. The last stretch asks: what happens when the objective itself is *learned from messy human judgments*, and how do you do that statistically without fooling yourself? That is the machinery of preference data, reward models, RLHF, and DPO, and it is where the book ends. Chapter 26 still holds the **map of where to go next** for breadth, and this chapter was the last large region on that map: statistics over time, where Chapter 3's independence assumption finally gets relaxed rather than assumed. Look back at where you started - sample means and spreads in Chapters 1-2 - and where you now stand: autocorrelation, martingale-style recursions, Kalman updating, and the linear dynamical systems inside modern sequence models. The same handful of tools - expectation, variance, correlation, likelihood, Bayes, and the Markov property - carried you the whole way.

<!-- chapter 36 -->