# Chapter 28: Diffusion Models and Score Matching - Statistics as Generation

## The idea this chapter builds toward

For the entire book so far, statistics has been about *summarizing* and *predicting* data - estimate a mean (Chapter 9), fit a line (Chapter 14), classify a label (Chapter 25). This chapter is about something that sounds much harder and turns out to be the same machinery: **generating** new data. Stable Diffusion, DALL-E, Imagen, and the image and video generators behind them all rest on one deceptively simple statistical trick: take a real image, add noise to it step by step until it is pure static, then train a model to **undo** one tiny noising step at a time. If you can reliably answer "which way, statistically, is less noisy?", you can run that answer backward from pure static and walk into a realistic image. Every ingredient is already yours: Chapter 6's Normal distribution, Chapter 24's Markov chains and Monte Carlo, Chapter 10's MLE, and Chapter 14's regression.

> **Intuition, in one breath:** Destroying data with noise is easy and fully known; the whole art is learning a small statistical model of how to go one step back, and applying that step over and over is how a generative model is born.

## Destroying data on purpose - the forward noising process

Start with a real data point $x_0$ (say, a pixel value, or a whole image flattened to a vector). Define a sequence $x_1, x_2, \ldots, x_T$ by adding a little Gaussian noise at each step:

$$x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\,\varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, I)$$

**Reading it piece by piece:**
- **$x_{t-1}$ and $x_t$** - the data (or noisy data) at the previous and current step; each step is built directly from the one before it, which is what makes this a Markov chain.
- **$\sqrt{1-\beta_t}$** - a shrunken copy of the old signal: because variance scales with the square of a coefficient (Chapter 7's $\text{Var}(aX)=a^2\text{Var}(X)$), this factor is chosen so the surviving signal keeps a controlled fraction of its variance.
- **$\sqrt{\beta_t}$** - the matching coefficient on the fresh noise, sized so signal variance plus noise variance stay constant - the two square roots together conserve total variance instead of letting it grow or collapse.
- **$\beta_t \in (0,1)$** - the variance schedule: a small per-step number deciding how much noise is injected at step $t$; small values keep each step gentle and its reverse easy to learn.
- **$\varepsilon_t$** - a fresh draw of standard Normal noise, independent at each step, representing the new randomness being mixed into the signal.

where $\beta_t \in (0,1)$ is the **variance schedule** - a small number controlling how much noise is injected at step $t$. This is exactly a **Markov chain** (Chapter 24): the distribution of $x_t$ depends only on $x_{t-1}$, not the whole history. The $\sqrt{1-\beta_t}$ factor (rather than a bare 1) is there to conserve signal variance - it is the same variance-scaling bookkeeping as Chapter 7's $\text{Var}(aX)=a^2\text{Var}(X)$, applied at every step so the chain neither blows up nor collapses.

Why bother with many tiny steps instead of one big jump to noise? Because the reverse of a *tiny* Gaussian nudge is itself approximately Gaussian and easy to model, whereas the reverse of a huge jump is a complicated, multimodal thing. Small steps buy tractability.

## A miracle of the Gaussian: jump straight to any step

Because Gaussians compose so cleanly (Chapter 6's nice property: sums of independent Gaussians are Gaussian), there's a closed form for $x_t$ directly in terms of $x_0$, skipping all intermediate steps. Writing $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,I)$$

**Reading it piece by piece:**
- **$x_0$** - the original clean data point; the jump formula expresses any step directly in terms of it, skipping all intermediate states.
- **$\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$** - the compounded retention of signal after all $t$ steps, where each $\alpha_s = 1-\beta_s$; it starts near 1 (little damage) and decays toward 0 as $t$ grows.
- **$\sqrt{\bar\alpha_t}$** - how much of the original signal survives in $x_t$; because variances add as squares, the coefficient is the square root of the retained fraction.
- **$\sqrt{1-\bar\alpha_t}$** - how much total noise has accumulated; it grows as the signal shrinks, and the pairing again conserves variance at every step.
- **$\varepsilon$** - one single standard Normal draw that stands in for all the accumulated noise; sampling just this one vector reproduces any step's noise in one line.
- **$\mathcal{N}(0,I)$** - the standard Normal distribution the noise is drawn from, known completely so the final state $x_T$ is trivially samplable.

This is the **reparameterization** trick, and it is enormously practical: to train on step $t$ you don't have to actually run $t$ noising steps - you just sample one $\varepsilon$ and compute $x_t$ in one line. As $t \to T$, $\bar\alpha_t \to 0$ and $x_T$ becomes essentially pure standard Normal noise, independent of the original image. That is the target state: a distribution we know completely, from which we can sample trivially (Chapter 24).

## Running it backward - the reverse denoising process

The generative dream: if we knew the reverse conditional $q(x_{t-1}\mid x_t)$, we could start from pure noise $x_T$ and sample $x_{T-1}, \ldots, x_0$, ending with a fresh realistic image. We don't know it - but we can *learn* an approximation $p_\theta(x_{t-1}\mid x_t)$. The 2020 DDPM paper (Ho, Jain, Abbeel) made a crucial empirical observation: for small $\beta_t$, the true reverse step is approximately Gaussian, so we only need to learn its **mean**, with a fixed variance. And here is the punchline - the optimal reverse mean depends on $x_t$ and on one unknown quantity: the **noise** $\varepsilon$ that was added. So instead of predicting the mean directly, train a network $\varepsilon_\theta(x_t, t)$ to predict the noise $\varepsilon$ that produced $x_t$. Prediction becomes **regression** in the most literal Chapter-14 sense: fit a function to a numeric target, minimizing squared error:

$$\mathcal{L} = \mathbb{E}_{t,\,x_0,\,\varepsilon}\Big[\,\|\,\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\ t)\,\|^2\,\Big]$$

**Reading it piece by piece:**
- **$\mathbb{E}_{t,\,x_0,\,\varepsilon}$** - the expectation averages over random timesteps $t$, real data $x_0$, and noise draws $\varepsilon$, so the loss reflects the whole training distribution rather than one example.
- **$\varepsilon$** - the true noise that was actually added to build $x_t$; this is the target the network is asked to recover.
- **$\varepsilon_\theta(\cdot,\ t)$** - the network's prediction of that noise, given the noisy input and which timestep it came from; $\theta$ are its learned weights.
- **$\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$** - the noisy $x_t$ fed to the network, built by the closed-form jump so no intermediate steps are simulated.
- **$\|\,\cdot\,\|^2$** - squared Euclidean distance, the error between predicted and true noise; minimizing it is exactly Chapter 14's regression and, against a Gaussian target, Chapter 10's likelihood maximization.

This is the entire training loss of DDPM. Note the deep connection to Chapter 10's MLE: minimizing squared error against Gaussian targets *is* maximizing likelihood (Chapter 25 re-derived exactly this). The diffusion loss is "predict-the-noise" regression, nothing more exotic.

<div class="labcard">
<div class="labkicker">Live - corrupt a 1-D signal and watch the reverse step undenoise it</div>
<div class="labcontrols">
  <div class="labctrl"><label>Timestep t <b id="ch28_t_label">0</b></label><input type="range" id="ch28_t" min="0" max="100" step="1" value="0"></div>
</div>
<div class="labrow">
  <button class="labbtn" id="ch28_noise">Add noise step</button>
  <button class="labbtn secondary" id="ch28_denoise">Denoise one step</button>
  <button class="labbtn secondary" id="ch28_reset">Reset signal</button>
</div>
<div class="labcanvas-wrap"><canvas id="ch28_canvas" width="820" height="240"></canvas></div>
<div class="labstat-grid">
  <div class="labstat"><div class="lbl">sqrt(alpha_bar)</div><div class="val" id="ch28_ab">-</div></div>
  <div class="labstat"><div class="lbl">SNR</div><div class="val" id="ch28_snr">-</div></div>
  <div class="labstat"><div class="lbl">Noise scale</div><div class="val" id="ch28_ns">-</div></div>
</div>
<p class="labnote">The blue curve is the clean signal x0; the orange curve is x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*noise. Raise t to watch the curve degrade into static; "Denoise one step" nudges t back down the way the reverse process walks from noise toward data.</p>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var SL = window.StatsLab;
  var N = 100;
  var x0 = [];
  for(var i=0;i<N;i++) x0.push(Math.sin(i/8) + 0.6*Math.exp(-Math.pow((i-50)/10,2)));
  var noisy = [1.2, -0.4, 0.7, -1.1, 0.3, 1.4, -0.8, 0.1, -1.3, 0.5];
  function ab(t){
    var frac = t/100;
    var logab = 0;
    for(var s=1;s<=100;s++){
      var beta = 0.0001 + 0.02*(s/100);
      logab += Math.log(1-beta)*(s<=t?frac:0);
    }
    var simple = Math.pow(0.02, frac*frac);
    return Math.max(1e-6, Math.min(1, simple));
  }
  function draw(){
    var t = parseInt(document.getElementById('ch28_t').value);
    var a = ab(t), sq = Math.sqrt(a), nsc = Math.sqrt(Math.max(0,1-a));
    document.getElementById('ch28_t_label').textContent = t;
    document.getElementById('ch28_ab').textContent = sq.toFixed(4);
    document.getElementById('ch28_ns').textContent = nsc.toFixed(4);
    var snr = a/(Math.max(1e-9,1-a));
    document.getElementById('ch28_snr').textContent = snr.toFixed(3);
    var xt = x0.map(function(v,i){ return sq*v + nsc*noisy[i%noisy.length]; });
    var plot = SL.makePlot(document.getElementById('ch28_canvas'));
    var xs = x0.concat(xt);
    var lo = Math.min.apply(null,xs), hi = Math.max.apply(null,xs);
    var pad = Math.max(0.5,(hi-lo)*0.1);
    plot.setDomain([0,N-1],[lo-pad,hi+pad]); plot.clear();
    plot.axes('signal index','value');
    plot.line(x0.map(function(v,i){return [i,v];}), SL.colors.accent, 2);
    plot.line(xt.map(function(v,i){return [i,v];}), SL.colors.accent2, 2);
  }
  document.getElementById('ch28_t').addEventListener('input', draw);
  document.getElementById('ch28_noise').addEventListener('click', function(){
    var el = document.getElementById('ch28_t');
    el.value = Math.min(100, parseInt(el.value)+10); draw();
  });
  document.getElementById('ch28_denoise').addEventListener('click', function(){
    var el = document.getElementById('ch28_t');
    el.value = Math.max(0, parseInt(el.value)-10); draw();
  });
  document.getElementById('ch28_reset').addEventListener('click', function(){
    document.getElementById('ch28_t').value = 0; draw();
  });
  draw();
});
</script>

## How this falls out of an ELBO

Faithful derivations (DDPM's original) arrive here through a variational bound - the **ELBO**, evidence lower bound, the same object behind VAEs in Chapter 23. Since directly maximizing the data likelihood $\log p_\theta(x_0)$ is intractable, we lower-bound it and maximize the bound instead, cross-entropy-style (Chapter 21). When you grind through the Gaussian algebra, the bound's dominant term collapses to precisely the squared-error noise-prediction loss above. That is *why* "predict the noise" is the right objective rather than an arbitrary choice: it is the parameterization that makes the intractable likelihood bound both calculable and numerically gentle.

## Score matching - the same idea seen from calculus

There's a second, equivalent lens worth having, because it connects straight to Chapter 10's derivative logic. The **score** of a density $p(x)$ is defined as the gradient of its log-density:

$$s(x) = \nabla_x \log p(x)$$

**Reading it piece by piece:**
- **$\nabla_x$** - the gradient taken with respect to the *data point* $x$, not any parameter; it asks "if I nudge this data point, which way does the density move?"
- **$\log p(x)$** - the log-density, the same log transform used throughout the book to turn products into sums and tame scales.
- **$s(x)$** - the resulting score vector at each point, which equals $\nabla_x p(x)/p(x)$ and so points in the direction of steepest increase in probability - uphill in density-space.

Why the log-gradient? Because $\nabla_x \log p(x) = \nabla_x p(x)/p(x)$ - it points in the direction of *greatest increase in probability*, uphill in density-space. In Chapter 10 we found the MLE by setting a derivative (of the log-likelihood) to zero; the score is that same derivative, but as a *function of the data rather than the parameters*. Sampling from a distribution can be done by climbing the score: **Langevin dynamics** (Chapter 24's MCMC family) iterates

$$x \leftarrow x + \tfrac{\eta}{2}\,\nabla_x \log p(x) + \sqrt{\eta}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,I)$$

**Reading it piece by piece:**
- **$x$** - the current sample being moved; each iteration replaces it with a slightly better one.
- **$\tfrac{\eta}{2}\,\nabla_x \log p(x)$** - the drift step: a fraction of the score, pushing the sample uphill toward higher-probability regions; $\eta$ sets the step size.
- **$\sqrt{\eta}\,\varepsilon$** - injected Gaussian noise, scaled so its variance grows in step with the drift (the $\sqrt{\eta}$ is the variance-conservation bookkeeping again); it keeps the walk exploratory rather than letting it collapse onto one peak.
- **$\varepsilon \sim \mathcal{N}(0,I)$** - a fresh standard Normal draw each iteration, the source of randomness in the walk.

which drifts $x$ uphill toward high-density regions while injecting noise to explore rather than collapse to a single peak - a stochastic version of gradient ascent on density.

The Trento/1999 insight (Hyvärinen) was that you can fit $s_\theta(x) \approx \nabla_x \log p(x)$ *without ever knowing $p(x)$*, via score matching. And it turns out the noise-prediction network is secretly a score network: for the Gaussian convention above, $s_\theta(x_t) \approx -\varepsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$. So "predict the noise" and "estimate the score" are the *same function*, viewed two ways. That equivalence is what links denoising diffusion to the older Langevin/score-based generative literature.

## The variance schedule, intuitively

The sequence $\{\beta_t\}$ decides how the model spends its capacity. Early steps (small $t$) remove fine texture; late steps remove global structure. A **linear** schedule races to noise too fast - little time is spent on the perceptually important coarse scales. The **cosine schedule** (Nichol & Dhariwal, 2021) smooths the transition so noise accumulates more gently, keeping $\bar\alpha_t$ from collapsing early, which measurably improves sample quality. Too-fast noising also makes the reverse step's Gaussian approximation poor (large jumps are less reversible), so the schedule is a genuine statistical modeling choice, not a cosmetic knob.

## Classifier-free guidance - steer by comparing two predictions

Modern text-to-image systems don't sample $p_\theta(x_t \mid \text{text})$ directly; they extrapolate between conditional and unconditional scores:

$$\hat\varepsilon = \varepsilon_\theta(x_t, \varnothing) + w\big(\varepsilon_\theta(x_t, c) - \varepsilon_\theta(x_t, \varnothing)\big)$$

**Reading it piece by piece:**
- **$\varepsilon_\theta(x_t, \varnothing)$** - the network's noise prediction with *no* condition (the empty prompt), the unconditional baseline of what noise looks like.
- **$\varepsilon_\theta(x_t, c)$** - the network's prediction conditioned on the text (or class) $c$; the difference from the baseline captures the conditioning's influence.
- **$\varepsilon_\theta(x_t, c) - \varepsilon_\theta(x_t, \varnothing)$** - the text's causal push: how much the condition shifts the noise prediction away from the unconditional one.
- **$w > 1$** - the guidance weight that amplifies that push; larger $w$ sharpens prompt adherence at the cost of diversity and sometimes artifacts - a bias-variance tradeoff like Chapter 16's regularization.
- **$\hat\varepsilon$** - the final guided noise estimate actually used to run the reverse step, an extrapolation beyond the conditional prediction.

with guidance weight $w > 1$. The bracket is the *difference* between "what noise looks like given the text" and "what it looks like given nothing" - the text's causal push, amplified. This is a variance/bias tradeoff in the same spirit as Chapter 16's regularization: higher $w$ sharpens prompt adherence (closer to the conditioning) at the cost of diversity and sometimes artifacts.

## Worked example - by hand, larger

Let's push one full training step of a diffusion model through by hand, with tiny numbers.

**Setup.** One scalar data point $x_0 = 2.0$. Use a schedule whose compounded values at step $t$ happen to be $\bar\alpha_t = 0.81$, so $\sqrt{\bar\alpha_t}=0.9$ and $\sqrt{1-\bar\alpha_t}=\sqrt{0.19}=0.4359$. Suppose training draws the noise $\varepsilon = 0.5$.

**Forward (corrupt).** Apply reparameterization: $x_t = 0.9\times 2.0 + 0.4359\times 0.5 = 1.8 + 0.2180 = 2.0180$. So the noisy input the network sees is $2.018$.

**Network prediction.** Pretend the network currently outputs $\hat\varepsilon = 0.30$ (it has not learned yet that the true noise was bigger).

**Loss.** Squared error: $\mathcal{L} = (0.5 - 0.30)^2 = 0.20^2 = 0.0400$.

**Gradient step.** For $\frac{\partial \mathcal{L}}{\partial \hat\varepsilon} = -2(\varepsilon - \hat\varepsilon) = -2(0.5-0.3) = -0.40$. With learning rate $\eta = 0.1$, the output updates by $-\eta \times (-0.40) = +0.040$, so $\hat\varepsilon \leftarrow 0.30 + 0.040 = 0.340$. Recompute loss: $(0.5-0.34)^2 = 0.16^2 = 0.0256$, down from $0.0400$. The step did exactly what regression training does - pulled the prediction toward the target.

**Recovering the reverse mean.** From the true relation $\varepsilon = (x_t - \sqrt{\bar\alpha_t}\,x_0)/\sqrt{1-\bar\alpha_t}$, the implied clean value is $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon)/\sqrt{\bar\alpha_t} = (2.0180 - 0.4359\times0.34)/0.9 = (2.0180-0.1482)/0.9 = 1.8698/0.9 = 2.0776$. Even from a half-learned noise estimate, denoising nudges us back toward the true $x_0=2.0$ - that single nudge, repeated $T$ times from pure noise, is generation.

**The takeaway:** training a diffusion model is just fitting a regression to predict added Gaussian noise, and each prediction is a small "how do I move away from noise" step - the whole generative engine is regression plus Markov-chain bookkeeping.

## Check your understanding

- Why does the reparameterization $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$ let training skip the intermediate steps, and which Chapter 6 property of Gaussians makes it valid?
- In what precise sense is "predict the noise $\varepsilon$" the same task as "estimate the score $\nabla_x\log p(x)$," and why does that link diffusion to Langevin sampling from Chapter 24?
- Why is a small per-step noise $\beta_t$ important for the reverse process to be learnable as a simple Gaussian, and what goes wrong with one large jump?

## Exercises

### Exercise 1 - A forward noising step, by hand
With $\bar\alpha_t = 0.64$ for a chosen step, data $x_0 = 3.0$, and sampled noise $\varepsilon = -1.0$, compute (a) $x_t$ and (b) $-\varepsilon/\sqrt{1-\bar\alpha_t}$, the corresponding score estimate using $s_\theta(x_t)\approx-\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$ with $\varepsilon_\theta=\varepsilon$.

<details><summary>Show solution</summary>

(a) $\sqrt{0.64}=0.8$, $\sqrt{1-0.64}=\sqrt{0.36}=0.6$. Then $x_t = 0.8\times3.0 + 0.6\times(-1.0) = 2.4 - 0.6 = 1.8$.

(b) $-\varepsilon/\sqrt{1-\bar\alpha_t} = -(-1.0)/0.6 = 1.0/0.6 = 1.667$. Positive score at $x_t=1.8$ means density is increasing in the $+x$ direction there, i.e. $x_t$ sits below the local peak and Langevin dynamics would drift it upward - consistent with the clean value $3.0$ being larger than the noisy $1.8$. This is Chapter 10's log-derivative idea: the score is the log-density's slope, and its sign tells you which way uphill is.

</details>

### Exercise 2 - Regression target, conceptually
A colleague says "diffusion models learn the data distribution directly." Correct this. What single scalar function does the network actually learn to output, and which Chapter 14 and Chapter 10 objects is that task?

<details><summary>Show solution</summary>

The network does **not** learn to output a density $p(x)$ directly - densities over high-dimensional images are intractable to represent. It learns to output the **noise vector** $\varepsilon_\theta(x_t,t)$ that was used to corrupt a clean input, i.e. it solves a mean-squared-error regression problem with a numeric target (Chapter 14). Because MSE against Gaussian targets is the negative-log-likelihood of a Gaussian (Chapter 10/25), the training objective is equivalently maximizing a variational lower bound on the data likelihood - but the *mechanism* the network performs is prediction of one sampled noise vector. The distribution is captured implicitly by the composition of many simple reverse steps, not by any single explicit density.

</details>

### Exercise 3 - Why small steps
Suppose you set $T=1$ with $\beta_1=0.99$, so one jump takes $x_0 \to x_1$ almost equal to pure noise. Explain why learning the reverse single step is much harder than learning $T=1000$ tiny steps, connecting to multimodality and the Gaussian reverse-step assumption.

<details><summary>Show solution</summary>

A single large jump maps *every* $x_0$ to nearly the same noise, so the reverse map $x_1 \to x_0$ must recover a whole complex, highly multimodal data distribution from close-to-universal noise in one shot - no single Gaussian or simple function approximates that. With many tiny steps, each forward step adds a *tiny* Gaussian perturbation, and a tiny Gaussian perturbation has a tiny, nearly Gaussian reverse (the true posterior $q(x_{t-1}\mid x_t)$ is approximately Gaussian when $\beta_t$ is small). Each reverse step therefore only needs to model a small, well-behaved neighborhood of the data manifold, which a simple learned mean handles well. This is the same "make each step locally simple, chain them" logic that makes small-step MCMC (Chapter 24) work, and it is why the variance schedule matters: too-fast noising violates the Gaussian reverse-step approximation the whole method depends on.

</details>

## Why This Matters - In AI and In Everyday Life

**In AI / machine learning:**

- **Stable Diffusion, DALL-E, Imagen, Midjourney, and modern video generators** are all denoising diffusion models with text conditioning - the noise-prediction regression and classifier-free guidance this chapter describes are literally their training and sampling loop.
- **The predict-the-noise loss is everywhere** in generative AI now, with the same MSE-vs-likelihood identity from Chapters 10/25 reused at billion-parameter scale on billions of images.
- **Score-based and flow-matching models** (including the rectified-flow / flow-matching objective behind several 2024-2026 frontier image and video models) are direct descendants of the score view: instead of estimating $\nabla_x\log p$ at a fixed noise level, they learn a velocity field transporting noise to data, but the statistical scaffolding is identical.
- **Diffusion policy in robotics and diffusion-based planners** reuse this exact framework for generating action sequences, showing that "generate by denoising" generalizes far beyond images.

**In everyday, non-AI life:**

- **A sculptor revealing a figure from marble**: the reverse process removes noise to reveal structure, and the artist's skill is knowing which tiny chip to make next - statistical denoising in physical form.
- **Restoring an old photograph**: gradual, stepwise cleanup (denoise, then de-blur, then sharpen) is how humans actually recover a degraded image, and it mirrors the reverse chain that generated models learn to run.

## What's next

Chapter 29 turns to the other dominant generative architecture of the modern era: the **transformer**. Where diffusion generates by reversing noise, transformers generate token by token using **attention** - and the surprising punchline is that attention itself is a soft, data-dependent weighted average that you can compute by hand, a topic we pick up next.