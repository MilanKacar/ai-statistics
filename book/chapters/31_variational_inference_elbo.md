# Chapter 31: Variational Inference and the ELBO - Approximate Bayes at Scale

## The idea this chapter builds toward

Chapter 17 gave you the Bayesian recipe: posterior ∝ likelihood × prior. Chapter 23 showed that once a model has latent variables, the exact likelihood turns into a log-of-a-sum that EM had to route around. This chapter confronts the general reason Bayesian inference is hard: computing the posterior $p(z\mid x)$ requires dividing by the **evidence** $p(x) = \int p(x\mid z)p(z)\,dz$, an integral over every possible value of the latent variables - which for a modern model with millions of latents is hopeless to compute exactly. **Variational inference (VI)** replaces that hopeless integral with an *optimization*: pick a family of tractable distributions $q_\phi(z)$ and find the member closest to the true posterior, measuring closeness with Chapter 21's KL divergence. The objective you optimize is the **ELBO** (Evidence Lower Bound), and it is the single loss function behind Variational Autoencoders, much of Bayesian deep learning, and the latent structure in diffusion models. This chapter is Chapter 23's latent-variable idea with Chapter 17's Bayesian precision, made computationally practical by Chapter 24's sampling.

> **Intuition, in one breath:** You can't compute the exact posterior, so you substitute a simple, tunable guess and keep nudging that guess until it sits as close as possible to the truth you can't see. The ELBO is the score that tells you how good your guess is, even though the truth is out of reach.

## Why the posterior is intractable (Chapter 4, Chapter 17)

Bayes' theorem for latent variables $z$ given data $x$ is

$$p(z\mid x) = \frac{p(x\mid z)\,p(z)}{p(x)}, \qquad p(x) = \int p(x\mid z)\,p(z)\,dz.$$

**Reading it piece by piece:**
- **$p(z\mid x)$** - the posterior over the latents $z$ given the observed data $x$; the full distribution we would like to know.
- **$p(x\mid z)$** - the likelihood: how well a candidate latent setting explains the observed data.
- **$p(z)$** - the prior: what we believed about the latents before seeing any data.
- **$p(x)$** - the evidence, the normalizer that makes the posterior a valid distribution.
- **$\int p(x\mid z)\,p(z)\,dz$** - the evidence written out as an integral over every possible latent value; in high dimensions this is the intractable bottleneck that motivates all of variational inference.

This is Chapter 4's theorem verbatim, with the normalizer now an integral over a high-dimensional latent space. Two separate problems conspire. First, the integral $p(x)$ rarely has a closed form - even in the simplest latent-variable models it is a log-of-a-sum or an intractable average, exactly Chapter 23's obstacle. Second, even if you could compute it, you'd still need $p(z\mid x)$ *itself* as a full distribution to do anything useful with it. For a neural network with millions of latent dimensions, both problems are fatal. Chapter 17's conjugate-prior trick and Chapter 24's MCMC are the escape hatches for small and medium problems, but MCMC scales poorly when each likelihood evaluation is a forward pass through a huge network and you need thousands of samples. VI takes the different route: don't approximate the *integral*, approximate the *posterior*.

## The variational family and KL closeness (Chapter 21)

Choose a family of distributions $q_\phi(z)$ parameterized by $\phi$ - for instance, a diagonal Gaussian where $\phi = (\mu, \sigma)$ for each latent dimension. This is your **variational family**, and it is deliberately *simple* so that sampling from it and evaluating it are cheap. We want the member of the family that is closest to the true posterior, and "closest" is measured by the **KL divergence** from Chapter 21:

$$D_{KL}(q_\phi(z)\,\|\,p(z\mid x)) = \mathbb{E}_{q_\phi}\!\left[\log\frac{q_\phi(z)}{p(z\mid x)}\right] \geq 0.$$

**Reading it piece by piece:**
- **$q_\phi(z)$** - the approximate posterior from our chosen variational family, parameterized by $\phi$.
- **$p(z\mid x)$** - the true posterior we are trying to match but cannot compute.
- **$\mathbb{E}_{q_\phi}[\cdot]$** - an expectation taken under the approximate posterior, averaging the log-ratio over its likely latent values.
- **$\log\frac{q_\phi(z)}{p(z\mid x)}$** - the log-ratio of guess to truth at each $z$; it is positive where the guess over-allocates probability and negative where it under-allocates.
- **$\geq 0$** - the divergence is never negative and is exactly zero only when the two distributions coincide, the Chapter 21 property that makes it a valid measure of closeness.

KL is non-negative - zero if and only if the two distributions coincide, exactly as Chapter 21 established. If we could minimize this directly, we'd be done. But the KL is defined *in terms of the very posterior $p(z\mid x)$ we cannot compute*. That circularity is the entire problem, and the ELBO is the clever way out.

## Deriving the ELBO - the bound that saves us (Chapter 21)

Start with the log-evidence and decompose it. Write $\log p(x)$ and multiply-and-divide inside the expectation by $q_\phi(z)$:

$$\log p(x) = \log \int p(x\mid z)p(z)\,dz = \log \mathbb{E}_{q_\phi}\!\left[\frac{p(x\mid z)p(z)}{q_\phi(z)}\right].$$

Apply Jensen's inequality (the log of an expectation is at least the expectation of the log, since $\log$ is concave - the same inequality Chapter 23's EM relied on):

$$\log p(x) \geq \mathbb{E}_{q_\phi}\!\left[\log\frac{p(x\mid z)p(z)}{q_\phi(z)}\right] = \text{ELBO}(\phi).$$

**Reading it piece by piece:**
- **$\log p(x)$** - the log-evidence, the quantity we wish we could maximize but cannot compute.
- **$\geq$** - the inequality comes from Jensen's, since the log of an expectation is at least the expectation of the log when $\log$ is concave; it certifies the right side never exceeds the target.
- **$\mathbb{E}_{q_\phi}[\cdot]$** - an expectation we can actually estimate by sampling from the simple approximate posterior.
- **$\log\frac{p(x\mid z)p(z)}{q_\phi(z)}$** - the log of likelihood-times-prior divided by the guess, with the guess acting as the importance weight that makes the expectation tractable.
- **$\text{ELBO}(\phi)$** - the resulting lower bound, so named because it sits below the log-evidence and depends only on evaluable quantities.

This lower bound is the **Evidence Lower Bound**. Its name is literal: it sits *below* the log-evidence we want, and it is computed *only* from quantities we can actually evaluate and sample. Now split the inside into two familiar pieces:

$$\text{ELBO}(\phi) = \underbrace{\mathbb{E}_{q_\phi}\!\left[\log p(x\mid z)\right]}_{\text{reconstruction}} - \underbrace{D_{KL}(q_\phi(z)\,\|\,p(z))}_{\text{KL to prior}}.$$

**Reading it piece by piece:**
- **$\text{ELBO}(\phi)$** - the single tractable objective we maximize over the variational parameters $\phi$.
- **$\mathbb{E}_{q_\phi}[\log p(x\mid z)]$ (reconstruction)** - how well latent samples explain the observed data, Chapter 10's expected log-likelihood; for a Gaussian decoder it is negative mean squared error, so this term wants latents that reconstruct well.
- **$D_{KL}(q_\phi(z)\|p(z))$ (KL to prior)** - how far the approximate posterior has strayed from the prior, Chapter 21's divergence acting as a regularizer.
- **the minus sign** - the two terms tug against each other: the reconstruction pulls latents toward the data while the KL pulls them back toward the prior, and $\phi$ finds the balance.

The first term is the **expected log-likelihood** - Chapter 10's object exactly: how well the latent samples explain the data, which for a Gaussian decoder is (up to constants) negative mean squared error, echoing Chapter 25. The second term is the **KL divergence to the prior** from Chapter 21: how far your approximate posterior has drifted from the prior $p(z)$, acting as a regularizer. Maximizing the ELBO means simultaneously reconstructing the data well *and* staying near the prior - the two terms pull against each other, and $\phi$ finds the balance.

## The exact gap is the KL we couldn't compute (Chapter 21)

The ELBO's deepest property: the missing piece between it and the log-evidence is *precisely* the KL divergence we wanted to minimize. A one-line rearrangement gives

$$\log p(x) = \text{ELBO}(\phi) + D_{KL}(q_\phi(z)\,\|\,p(z\mid x)).$$

**Reading it piece by piece:**
- **$\log p(x)$** - the true log-evidence, which is fixed and does not depend on our choice of $\phi$.
- **$\text{ELBO}(\phi)$** - the tractable bound we actually maximize.
- **$D_{KL}(q_\phi(z)\|p(z\mid x))$** - the exact gap between the bound and the truth, which is precisely the KL to the true posterior we could not compute directly.
- **the identity** - because the left side is constant and the gap is non-negative, maximizing the ELBO is mathematically identical to minimizing that unknowable KL, which is why optimizing the proxy works.

Since the left side doesn't depend on $\phi$ and the KL term is $\geq 0$, maximizing the ELBO is *identical* to minimizing $D_{KL}(q_\phi\|p(z\mid x))$ - the thing we couldn't compute directly. **The ELBO is a proxy whose optimization automatically squeezes the uncomputable KL toward zero.** And because the gap is non-negative, the ELBO can never exceed the true log-evidence, giving a rigorous sense of how much room for improvement remains. This is why VI is a *bound-based* approximation: it is not sampling the posterior, it is fitting the best simple stand-in and reporting how tight the fit is.

## The reparameterization trick - making the gradient flow (Chapter 6)

To optimize $\phi$ by gradient ascent we need $\nabla_\phi \,\mathbb{E}_{q_\phi}[\cdot]$, but the expectation is taken *with respect to $\phi$ itself*, so you can't just push the gradient inside the expectation the way you could in Chapter 30 when the *reward*, not the distribution, carried the dependence. The **reparameterization trick** fixes this. For a Gaussian $q_\phi(z) = N(\mu, \sigma^2)$ - the distribution whose two parameters you met in Chapter 6 - rewrite the sample not as $z \sim N(\mu, \sigma^2)$ but as

$$z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim N(0, I).$$

**Reading it piece by piece:**
- **$z$** - a sample from the approximate posterior, rewritten as a deterministic function of the parameters rather than a draw that hides its dependence on them.
- **$\mu$** - the posterior mean, a learnable parameter; shifting it shifts every sample.
- **$\sigma$** - the posterior standard deviation, also learnable; scaling it stretches or shrinks the spread of samples.
- **$\odot$** - element-wise multiplication, applying a separate $\sigma$ to each latent dimension.
- **$\epsilon \sim N(0, I)$** - the source of randomness, a standard Normal draw that carries no $\phi$ dependence, so the gradient can pass cleanly through $\mu$ and $\sigma$.

Now the randomness lives entirely in $\epsilon$, which carries *no* $\phi$ dependence, and $z$ is a deterministic, differentiable function of $\phi = (\mu,\sigma)$. The expectation is over the fixed $\epsilon$, so the gradient moves inside cleanly: $\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{\epsilon}[\nabla_\phi f(\mu + \sigma\odot\epsilon)]$ - a low-variance estimator, in contrast to the much noisier score-function estimator of Chapter 30. That $\mu + \sigma\epsilon$ form is Chapter 6's Normal distribution, restated so its parameters become differentiable knobs.

<div class="labcard">
<div class="labkicker">Live - balance reconstruction against the KL penalty</div>
<div class="labcontrols">
  <div class="labctrl"><label>Posterior mean &mu;q <b id="ch31_mu_label">1.5</b></label><input type="range" id="ch31_mu" min="-1" max="4" step="0.05" value="1.5"></div>
  <div class="labctrl"><label>Posterior std &sigma;q <b id="ch31_sigma_label">1.0</b></label><input type="range" id="ch31_sigma" min="0.2" max="3" step="0.05" value="1.0"></div>
  <div class="labctrl"><label>Prior std &sigma;0 <b id="ch31_prior_label">1.5</b></label><input type="range" id="ch31_prior" min="0.3" max="3" step="0.05" value="1.5"></div>
</div>
<div class="labcanvas-wrap"><canvas id="ch31_canvas" width="820" height="240"></canvas></div>
<div class="labstat-grid">
  <div class="labstat"><div class="lbl">Reconstruction log p(x|z)</div><div class="val" id="ch31_recon">-</div></div>
  <div class="labstat"><div class="lbl">KL(q||prior)</div><div class="val" id="ch31_kl">-</div></div>
  <div class="labstat"><div class="lbl">ELBO = recon - KL</div><div class="val" id="ch31_elbo">-</div></div>
</div>
<p class="labnote">The observed point is x = 2.0. Drag &mu;q toward x to raise the reconstruction term, but watch the KL penalty grow as the posterior drifts away from the prior N(0, &sigma;&#8320;&sup2;). The ELBO is the tug-of-war between "explain the data" and "stay near the prior" - the same tradeoff regularization expresses in Chapter 16.</p>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var SL = window.StatsLab;
  var X_OBS = 2.0;
  function update(){
    var mu = parseFloat(document.getElementById('ch31_mu').value);
    var sig = parseFloat(document.getElementById('ch31_sigma').value);
    var prior = parseFloat(document.getElementById('ch31_prior').value);
    document.getElementById('ch31_mu_label').textContent = mu.toFixed(2);
    document.getElementById('ch31_sigma_label').textContent = sig.toFixed(2);
    document.getElementById('ch31_prior_label').textContent = prior.toFixed(2);

    var recon = Math.log(SL.normalPDF(X_OBS, mu, sig) + 1e-12);
    var kl = 0.5 * ((mu*mu + sig*sig)/(prior*prior) - 1 - 2*Math.log(sig/prior));
    var elbo = recon - kl;

    document.getElementById('ch31_recon').textContent = recon.toFixed(3);
    document.getElementById('ch31_kl').textContent = kl.toFixed(3);
    document.getElementById('ch31_elbo').textContent = elbo.toFixed(3);

    var lo = Math.min(-3, mu-3*sig, -3*prior), hi = Math.max(5, mu+3*sig, 3*prior);
    var plot = SL.makePlot(document.getElementById('ch31_canvas'), {l:50,r:20,t:16,b:40});
    var priorPts = [], postPts = [];
    for(var i=0;i<=200;i++){
      var z = lo + i/200*(hi-lo);
      priorPts.push([z, SL.normalPDF(z, 0, prior)]);
      postPts.push([z, SL.normalPDF(z, mu, sig)]);
    }
    var maxY = Math.max(0.001, Math.max.apply(null, priorPts.map(function(p){return p[1];}).concat(postPts.map(function(p){return p[1];}))));
    plot.setDomain([lo,hi],[0,maxY*1.15]); plot.clear(); plot.axes('latent z','density');
    plot.line(priorPts, SL.colors.accent2, 2);
    plot.line(postPts, SL.colors.accent, 2);
    plot.vline(X_OBS, SL.colors.accent4, true);
  }
  ['ch31_mu','ch31_sigma','ch31_prior'].forEach(function(id){
    document.getElementById(id).addEventListener('input', update);
  });
  update();
});
</script>

## This is the objective behind VAEs and diffusion (Chapter 23, Chapter 28)

A **Variational Autoencoder** is exactly this machinery with neural networks: an encoder network outputs $(\mu_\phi, \sigma_\phi)$ for each latent dimension (the variational family), a decoder network models $p(x\mid z)$ (the reconstruction term), and the prior is a standard $N(0,I)$. The training loss is the negative ELBO - reconstruction error minus (or plus, after sign flip) the KL to the prior, which is why you'll see VAE losses written as "reconstruction + KL" and wondered where the KL came from: it is Chapter 21's divergence, serving as the term that keeps the learned latent codes spread out and near the prior instead of collapsing into a memorized table. This is the neural-network replacement for Chapter 23's EM E-step, doing the same "fit the approximate posterior" job without needing a closed form. **Diffusion models (Chapter 28)** share the skeleton: each denoising step is trained by maximizing a likelihood-style objective that, like the ELBO, decomposes into per-step reconstruction terms, and the variational view is what justifies treating the whole forward-noising/reverse-denoising process as one big latent-variable model fit by a bound-based objective.

## Worked example - by hand, larger

Let's compute every ELBO term by hand for the simplest possible case: a one-dimensional latent variable with a Gaussian prior and a Gaussian approximate posterior, and a single observed data point. This makes each piece arithmetic.

**Set-up.** Prior: $p(z) = N(0,1)$, so $\mu_0 = 0$, $\sigma_0^2 = 1$. Variational family: $q_\phi(z) = N(\mu, \sigma^2)$ with current parameters $\mu = 2$, $\sigma = 1$ (variance $\sigma^2 = 1$). Likelihood: given a latent $z$, the data is $p(x\mid z) = N(z, 1)$ - i.e. the decoder simply predicts $z$ with unit noise. Observed single data point: $x = 3$.

**Step 1 - the reconstruction term $\mathbb{E}_{q}[\log p(x\mid z)]$.** Under $p(x\mid z) = N(z,1)$, dropping the constant $\log(\sqrt{2\pi})$ (it doesn't involve $\phi$), $\log p(x\mid z) = -\frac{(x-z)^2}{2}$. Taking the expectation over $z \sim N(\mu=2, \sigma^2=1)$: by Chapter 7, $\mathbb{E}[(x-z)^2] = (x-\mu)^2 + \sigma^2 = (3-2)^2 + 1 = 1 + 1 = 2$. So the reconstruction term is $-\frac{1}{2}\times 2 = -1.0$.

**Step 2 - the KL-to-prior term $D_{KL}(q\|p)$.** For two univariate Gaussians with diagonal (here scalar) form, the closed form is

$$D_{KL}(q\|p) = \log\frac{\sigma_0}{\sigma} + \frac{\sigma^2 + (\mu-\mu_0)^2}{2\sigma_0^2} - \frac{1}{2}.$$

**Reading it piece by piece:**
- **$D_{KL}(q\|p)$** - the divergence between the approximate posterior $q$ and the prior $p$, both Gaussian; it measures how far the fit has drifted.
- **$\log\frac{\sigma_0}{\sigma}$** - the spread-mismatch term: it penalizes a $q$ whose standard deviation $\sigma$ differs from the prior's $\sigma_0$, and blows up if $\sigma$ shrinks toward zero.
- **$\frac{(\mu-\mu_0)^2}{2\sigma_0^2}$** - the location-mismatch term: a squared penalty on the posterior mean $\mu$ being far from the prior mean $\mu_0$, weighted by the prior's precision.
- **$\frac{\sigma^2}{2\sigma_0^2}$** - a correction for the posterior's own variance, keeping the expression exact for Gaussians.
- **$-\frac{1}{2}$** - a constant that makes the divergence exactly zero when $q$ and $p$ coincide.

Plug in $\mu=2$, $\sigma=1$, $\mu_0=0$, $\sigma_0=1$: $\log\frac{1}{1} = 0$; $\frac{1 + (2-0)^2}{2\times1} = \frac{1+4}{2} = 2.5$; minus $0.5$ gives $2.5 - 0.5 = 2.0$. So $D_{KL} = 2.0$.

**Step 3 - assemble the ELBO.** ELBO $=$ reconstruction $-$ KL $= -1.0 - 2.0 = -3.0$. That's our current lower bound on $\log p(x)$.

**Step 4 - improve $\phi$ and watch the ELBO rise.** The reconstruction term wants $\mu$ near the data $x=3$ (to reduce $(3-\mu)^2$) and the KL term wants $\mu$ near $0$ and $\sigma$ near $1$ (to stay close to the prior). Try the competing setting $\mu = 2.5$, $\sigma = 1$. Reconstruction: $\mathbb{E}[(3-z)^2] = (3-2.5)^2 + 1 = 0.25 + 1 = 1.25$, so that term is $-0.625$ (better). KL: $\frac{1 + (2.5)^2}{2} - 0.5 = \frac{1+6.25}{2} - 0.5 = 3.625 - 0.5 = 3.125$ (worse - $\mu$ moved further from the prior). New ELBO $= -0.625 - 3.125 = -3.75$, which is *lower*. So moving $\mu$ toward the data alone is not automatically good - the KL penalty fights back.

**Step 5 - find the winning balance.** The optimum trades off the two terms. The KL-minimizing $\mu$ is $0$; the reconstruction-minimizing $\mu$ is $3$. The ELBO's pull on the posterior mean is a precision-weighted compromise (echoing Chapter 17): with prior variance $1$, likelihood variance $1$, and one data point, the balance sits at $\mu \approx 1.5$, $\sigma \approx 1$. Check $\mu = 1.5$, $\sigma=1$: reconstruction $= -\frac12[(3-1.5)^2 + 1] = -\frac12[2.25+1] = -1.625$; KL $= \frac{1 + 2.25}{2} - 0.5 = 1.625 - 0.5 = 1.125$; ELBO $= -1.625 - 1.125 = -2.75$, higher than both $-3.0$ and $-3.75$.

**Takeaway: the ELBO is a tug-of-war between fitting the data (reconstruction, Chapter 10) and staying near the prior (KL, Chapter 21), and maximizing it is mathematically identical to minimizing the true posterior KL we could never compute directly - that identity is the whole reason variational inference works.**

## Check your understanding

- Why does the evidence integral $p(x) = \int p(x\mid z)p(z)\,dz$ make the exact posterior uncomputable, and how does replacing the *integral* with an *optimization over $q_\phi$* sidestep that?
- Given $\log p(x) = \text{ELBO}(\phi) + D_{KL}(q_\phi\|p(z\mid x))$ and the fact that KL is non-negative (Chapter 21), explain why maximizing the ELBO cannot push the approximate posterior *away* from the true one.
- What breaks if you try to backpropagate a gradient through a direct sample $z \sim N(\mu,\sigma^2)$, and how does $z = \mu + \sigma\epsilon$ fix it (Chapter 6)?

## Exercises

### Exercise 1 - Compute an ELBO by hand
A 1-D latent model has prior $p(z) = N(0,1)$ and a Gaussian likelihood $p(x\mid z) = N(z,1)$ with constant dropped. The variational posterior is $q(z) = N(\mu=1, \sigma=1)$, and the observed data point is $x = 2$. Compute both ELBO terms and the total.

<details><summary>Show solution</summary>

Reconstruction: $\mathbb{E}_{q}[(x-z)^2]/2$ with the negative sign, and $\mathbb{E}[(x-z)^2] = (x-\mu)^2 + \sigma^2 = (2-1)^2 + 1 = 1 + 1 = 2$. So reconstruction $= -\frac{1}{2}(2) = -1.0$.

KL to prior, using $D_{KL} = \log\frac{\sigma_0}{\sigma} + \frac{\sigma^2 + (\mu-\mu_0)^2}{2\sigma_0^2} - \frac12$ with $\sigma_0=1,\mu_0=0$: $\log 1 + \frac{1 + 1}{2} - 0.5 = 0 + 1 - 0.5 = 0.5$.

ELBO $= -1.0 - 0.5 = -1.5$.

Interpretation: because $\mu=1$ sits exactly midway between the prior's $\mu_0=0$ and the data point $x=2$, both terms are mild - the approximation is a reasonable compromise, and $-1.5$ is its certified lower bound on $\log p$ for this data point.

</details>

### Exercise 2 - Why the KL term prevents posterior collapse
In a VAE, suppose the encoder learns to output $\mu = 0$, $\sigma \to 0$ (an almost-point mass at the prior mean) for *every* input. Using the ELBO's KL term, explain why this degenerate solution is penalized, and what role the reconstruction term plays in preventing it from being optimal.

<details><summary>Show solution</summary>

If $q(z\mid x) = N(\mu=0, \sigma\to 0)$, substitute into the KL: $\log\frac{\sigma_0}{\sigma} + \frac{\sigma^2 + (\mu-\mu_0)^2}{2\sigma_0^2} - \frac12 = \log\frac{1}{\sigma} + \frac{\sigma^2}{2} - \frac12$. As $\sigma \to 0$, the $\log(1/\sigma)$ term blows up to $+\infty$ - so a collapsing posterior with vanishing variance is *heavily* penalized by the KL term. The divergence forbids the encoder from collapsing to a point mass.

But note the tension: a *moderately* shrunken $\sigma$ and $\mu \to 0$ *does* reduce the KL (pushing it toward zero). What stops the encoder from ignoring the input entirely and setting $\mu=0,\sigma=1$ (exactly the prior, KL $=0$) is the **reconstruction** term: if $z$ carries no information about $x$, the decoder cannot reconstruct $x$, and $\mathbb{E}[\log p(x\mid z)]$ becomes terrible. So reconstruction *forces* the latent code to retain information, while the KL term keeps it regularized and prevents collapse - the two terms jointly rule out both degenerate outcomes (ignoring the input, or cramming to a point mass). This is exactly the reconstruction/KL tug-of-war from the worked example, now explaining a real training pathology.

</details>

### Exercise 3 - Conceptual: VI versus MCMC and versus EM
Contrast variational inference with Chapter 24's MCMC and Chapter 23's EM on two axes: (a) what each one approximates (a distribution or a point), and (b) the bias-variance character of each.

<details><summary>Show solution</summary>

(a) *What is approximated.* MCMC (Chapter 24) targets the *true posterior* directly and asymptotically - with infinite samples it is exact, returning actual draws from $p(z\mid x)$. VI instead fits a member of a restricted family $q_\phi$; even with perfect optimization it is limited by the family's expressiveness, so it returns the *closest simple distribution*, generally not the true one. EM (Chapter 23) is different again: it finds a *point estimate* of the parameters (a single best $\theta$) by iterating, rather than a full distribution; VI returns a whole approximate posterior over latents.

(b) *Bias-variance character.* MCMC is asymptotically unbiased but high-variance and slow - you need many samples, and the samples are correlated, so it can be expensive and its answers noisy per unit compute. VI is biased (limited by the variational family, and prone to underestimating posterior variance since it fits a simple unimodal shape) but low-variance and fast - one forward pass through the encoder gives a fit, and it scales to huge neural models. EM is a middle case: it is a local optimizer (so it can land in a local optimum, a form of bias) but has no sampling variance at all - deterministic iterations. The practical rule of thumb: use VI when you need speed and scale and can tolerate a biased, somewhat under-dispersed posterior; use MCMC when you need trustworthy, near-exact posterior uncertainty for a smaller problem.

</details>

## Why This Matters - In AI and In Everyday Life

**In AI / machine learning:**

- **Variational Autoencoders**: the VAE training objective *is* the negative ELBO, with the encoder as the variational family and the KL-to-prior term as what keeps latent codes well-behaved - every generated-image or interpolation VAE is running this chapter.
- **Bayesian neural networks and uncertainty estimation**: keeping a distribution over network weights via VI (instead of a single point estimate) is how many systems produce calibrated uncertainty, connecting directly to Chapter 25's calibration concerns and Chapter 17's priors.
- **Diffusion and other generative latent-variable models**: the per-step training objectives in diffusion models (Chapter 28) are variational-style bounds over a hierarchy of latents, so the ELBO decomposition underpins the loss you actually optimize.
- **Large-scale approximate inference in production**: anywhere an exact posterior is too expensive - topic models, deep latent-variable recommenders, probabilistic embeddings - VI's optimization-instead-of-integration trick is the workhorse.

**In everyday, non-AI life:**

- **Best-fit simplifications of complicated beliefs**: when you can't reason exactly about a messy, multi-factor situation, you consciously adopt a simpler mental model and tune it until it matches your observations - that is fitting a variational family to an intractable posterior, done in your head.
- **Weather forecasting with a simplified model**: meteorologists run a tractable approximation of an impossibly complex atmospheric posterior; the goodness-of-fit bound they monitor is a practical cousin of the ELBO, telling them how much their simplification is costing them.

## What's next

Chapter 32 moves to **Contrastive Learning and InfoNCE**, where instead of matching distributions to a posterior or maximizing a reward, models learn representations by pulling matching pairs together and pushing mismatched pairs apart - an objective that turns out to be another information-theoretic bound, tying Chapter 21's mutual information to the embedding spaces behind modern retrieval and vision-language models.