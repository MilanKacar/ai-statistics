# Chapter 38: RLHF, DPO, and Preference Optimization - Turning Preferences Into a Loss Function

## The idea this chapter builds toward

Chapter 37 gave us a reward model: a learned, noisy scalar that predicts human preference. The classic way to use it - **RLHF** - is to run a policy-gradient algorithm (Chapter 30's PPO) to push a language model toward high-reward text, with a KL leash (Chapter 21) tying it to a reference model so it doesn't chase the reward model into gibberish. That works, but it's a complicated, unstable, three-model juggling act. This chapter's payoff is a genuinely beautiful statistical result: the KL-regularized reward-maximization objective has a **closed-form optimal policy**, and inverting that formula lets you express the reward *in terms of the policy itself* - which turns RLHF into a plain **binary-classification loss on preference pairs**, with **no reward model and no RL loop**. That method is **Direct Preference Optimization (DPO)**, and understanding *why* it works is a masterclass in using Chapter 21's KL divergence and Chapter 10's log-likelihood to collapse a hard reinforcement problem into an easy supervised one.

> **Intuition, in one breath:** The best policy for a KL-constrained reward is just the reference policy reweighted by how good each answer is, and that relationship can be flipped - so instead of learning a reward and then doing reinforcement learning, you can directly train the model to make preferred answers more likely than rejected ones.

## Recap: the RLHF pipeline (Chapters 30, 37, 21)

RLHF, the technique behind InstructGPT and most aligned chat models, assembles three earlier chapters into one loop:

1. **Collect comparisons.** Humans pick the preferred response from pairs, producing the preference data of Chapter 37.
2. **Fit a reward model** $r_\phi(x,y)$ via Bradley-Terry MLE (Chapters 37 and 10) - a learned scorer for "preferred-ness" given prompt $x$ and response $y$.
3. **Run PPO** (Chapter 30's clipped policy gradient) on the language model policy $\pi_\theta(y\mid x)$ to maximize the reward, with a **KL penalty** $-\beta\,D_{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$ toward the frozen pretrained reference $\pi_{\mathrm{ref}}$ (Chapter 21) to prevent the reward-hacking collapse of Chapter 27.

The full objective RLHF maximizes is

$$\max_{\pi_\theta}\ \mathbb{E}_{x,\,y\sim\pi_\theta}\Big[\, r_\phi(x,y) - \beta\, D_{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\big)\Big].$$

**Reading it piece by piece:**
- **$\pi_\theta$** - the language-model policy we are optimizing, the thing that decides which responses to produce.
- **$\mathbb{E}_{x,\,y\sim\pi_\theta}$** - the average over prompts $x$ and responses $y$ drawn from the current policy.
- **$r_\phi(x,y)$** - the learned reward model's score for a response, pushing toward text the reward model likes.
- **$\beta$** - the leash strength, trading off reward against staying close to the reference.
- **$D_{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$** - the divergence from the frozen reference policy, penalizing drift into reward-hacked gibberish.

The reward term says "produce text the reward model likes"; the KL term says "don't drift too far from the sane starting model." The trouble is that this is a full RL problem: sample from a moving policy, estimate advantages (Chapter 7), tune PPO's hyperparameters - expensive, unstable, and finicky when you launch three models (policy, reference, reward) at once.

## The key derivation - the KL-regularized objective has a closed-form optimal policy

Here is the statistical gem, and it's worth following every line. Fix a prompt $x$ and, for that fixed $x$, treat the whole response distribution as the thing we're optimizing. The objective decouples across prompts, so we can solve a *single* one. Write the per-prompt objective over distributions $p = \pi_\theta(\cdot\mid x)$:

$$\max_{p}\ \sum_y p(y)\, r(x,y) - \beta \sum_y p(y)\log\frac{p(y)}{\pi_{\mathrm{ref}}(y)}.$$

This is a constrained optimization: maximize expected reward minus $\beta$ times a KL divergence, subject to $\sum_y p(y) = 1$. Use a Lagrange multiplier $\lambda$ for the constraint and take the derivative with respect to each $p(y)$ (differentiating $p\log p$ using Chapter 21's entropy form):

$$r(x,y) - \beta\Big(\log\frac{p(y)}{\pi_{\mathrm{ref}}(y)} + 1\Big) + \lambda = 0.$$

Solve for $p(y)$ - the $+1$ and $\lambda$ are constants absorbed into a normalizer - giving the **closed-form optimal policy**

$$p(y) \propto \pi_{\mathrm{ref}}(y)\,e^{r(x,y)/\beta}, \qquad \pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big),$$

**Reading it piece by piece:**
- **$\pi^*(y\mid x)$** - the optimal policy, the best response distribution for a given prompt.
- **$\pi_{\mathrm{ref}}(y\mid x)$** - the frozen reference policy, the sane starting point being reweighted.
- **$e^{r(x,y)/\beta}$** - an exponential tilt by the reward, multiplying up the probability of high-reward responses.
- **$\beta$** - the temperature; small $\beta$ sharpens the tilt toward reward, large $\beta$ keeps it close to the reference.
- **$Z(x)$** - the partition function normalizing the reweighted distribution so it sums to one over all responses.

where $Z(x) = \sum_y \pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}$ is a normalizing constant (a partition function). Read it plainly: **the optimal policy is the reference policy reweighted by an exponential in the reward.** High-reward responses get their probability multiplied up by $e^{r/\beta}$; the temperature $\beta$ controls how aggressively (small $\beta$ = sharper tilt toward reward, exactly like low temperature in sampling, Chapter 21). This is a softmax-shaped blow-up of the prior - the same "prior times likelihood, renormalized" structure as Bayesian updating (Chapter 17), with the reward playing the likelihood's role.

## Inverting the formula - the reward in terms of the optimal policy

That closed-form is pretty but not yet directly useful, because $Z(x)$ is a sum over all possible responses - intractable for language. The trick is to **invert** it. Take logs of both sides and solve for the reward:

$$\log \pi^*(y\mid x) = \log \pi_{\mathrm{ref}}(y\mid x) + \tfrac{1}{\beta}r(x,y) - \log Z(x) \;\implies\; r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x).$$

**Reading it piece by piece:**
- **$\log \pi^*(y\mid x)$** - the log-probability the optimal policy assigns to response $y$.
- **$\log \pi_{\mathrm{ref}}(y\mid x)$** - the log-probability under the reference, the baseline being shifted.
- **$\tfrac{1}{\beta}r(x,y)$** - the reward contribution, scaled by the inverse temperature.
- **$-\log Z(x)$** - the normalizer as a constant off to the side.
- **$\beta\log\frac{\pi^*}{\pi_{\mathrm{ref}}}$** - the inverted form: the reward expressed purely as a scaled log-ratio of policy to reference.

Now plug this into the Bradley-Terry preference probability (Chapter 37) that $y_w$ beats $y_l$, namely $\sigma\big(r(x,y_w) - r(x,y_l)\big)$. The magic: **the $Z(x)$ terms cancel**, because both rewards share the same prompt-dependent normalizer:

$$r(x,y_w) - r(x,y_l) = \beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.$$

The intractable partition function vanishes - the *reward difference*, which is all Bradley-Terry ever needed, depends only on the policy and reference, both of which we can evaluate. This substitution is the entire insight behind DPO.

## The DPO loss - Bradley-Terry with the policy standing in for the reward

We aren't handed the optimal policy $\pi^*$; we're learning it as $\pi_\theta$. So replace $\pi^*$ by $\pi_\theta$ and fit $\theta$ by maximum likelihood on the preference data - Chapter 10 one more time. For a dataset of pairs $(x, y_w, y_l)$ where $y_w$ is preferred, the **DPO loss** (negate the log-likelihood so we're minimizing) is

$$\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\!\Big(\beta\Big[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\Big]\Big)\right].$$

**Reading it piece by piece:**
- **$\mathcal{L}_{\text{DPO}}(\theta)$** - the loss we minimize over the policy parameters, the negative log-likelihood of the preferences.
- **$\mathbb{E}_{(x,y_w,y_l)}$** - the average over preference triples: a prompt, a winner, and a loser.
- **$\sigma(\cdot)$** - the sigmoid turning the reward gap into a probability that the winner is preferred.
- **$\beta$** - the scaling that sets how sharply a reward gap translates into preference odds.
- **$\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}$** - the winner's log-ratio of policy to reference, and likewise for the loser, so their difference is the implicit reward gap.
- **$-\log \sigma$** - negating the log-probability makes this exactly logistic / cross-entropy loss, low when the winner is ranked above the loser.

Read it term by term:

- $\log \pi_\theta(y_w\mid x) - \log \pi_\theta(y_l\mid x)$ - the log-probability gap the model currently assigns to the winner over the loser. This is a **log-likelihood** (Chapter 10) ratio; recall Chapter 21 that $\log p - \log q = \log(p/q)$ is a KL-flavored divergence between the winner and loser distributions.
- $-\log \pi_{\mathrm{ref}}(y_w\mid x) + \log \pi_{\mathrm{ref}}(y_l\mid x)$ - the same gap under the frozen reference. Subtracting it gives the *change* in log-odds relative to where we started - a KL-like drift measure (Chapter 21), the "leash" showing up inside the loss instead of in an RL reward.
- The whole bracket is the **implicit reward difference** from the inversion above, scaled by $\beta$. Passing it through $\sigma$ and taking $-\log$ makes it **logistic / cross-entropy loss** (Chapters 25 and 10): we are literally training a binary classifier where the two classes are "winner" and "loser," and the logits are the policy-to-reference log-ratios.

That's the trick: DPO *is* logistic regression on preference pairs, marketed as RLHF - no reward model to train, no PPO sampling loop, no advantage estimation. One forward pass on $y_w$ and $y_l$, one backward pass.

<div class="labcard">
<div class="labkicker">Live - move the log-probs and watch the DPO loss respond</div>
<div class="labcontrols">
  <div class="labctrl"><label>log &pi;&theta;(winner) <b id="ch38_lpw_label">-2.0</b></label><input type="range" id="ch38_lpw" min="-6" max="-0.1" step="0.1" value="-2.0"></div>
  <div class="labctrl"><label>log &pi;&theta;(loser) <b id="ch38_lpl_label">-3.0</b></label><input type="range" id="ch38_lpl" min="-6" max="-0.1" step="0.1" value="-3.0"></div>
  <div class="labctrl"><label>log &pi;ref(winner) <b id="ch38_lrw_label">-2.1</b></label><input type="range" id="ch38_lrw" min="-6" max="-0.1" step="0.1" value="-2.1"></div>
  <div class="labctrl"><label>log &pi;ref(loser) <b id="ch38_lrl_label">-2.6</b></label><input type="range" id="ch38_lrl" min="-6" max="-0.1" step="0.1" value="-2.6"></div>
  <div class="labctrl"><label>&beta; <b id="ch38_beta_label">0.10</b></label><input type="range" id="ch38_beta" min="0.01" max="0.5" step="0.01" value="0.1"></div>
</div>
<div class="labcanvas-wrap"><canvas id="ch38_canvas" width="820" height="220"></canvas></div>
<div class="labstat-grid">
  <div class="labstat"><div class="lbl">Implicit reward (winner)</div><div class="val" id="ch38_rw">-</div></div>
  <div class="labstat"><div class="lbl">Implicit reward (loser)</div><div class="val" id="ch38_rl">-</div></div>
  <div class="labstat"><div class="lbl">&sigma;(reward diff)</div><div class="val" id="ch38_sig">-</div></div>
  <div class="labstat"><div class="lbl">DPO loss = -log &sigma;</div><div class="val" id="ch38_loss">-</div></div>
</div>
<p class="labnote">The implicit reward is &beta; &times; (log &pi;&theta; - log &pi;ref) for each response. Slide the winner's policy log-prob up (making &pi;&theta; like the winner more than &pi;ref did) and watch &sigma;(reward diff) climb toward 1 and the DPO loss drop toward 0 - this is logistic regression doing its job on preference pairs.</p>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var SL = window.StatsLab;
  function sigmoid(x){ return 1/(1+Math.exp(-x)); }
  function update(){
    var lpw = parseFloat(document.getElementById('ch38_lpw').value);
    var lpl = parseFloat(document.getElementById('ch38_lpl').value);
    var lrw = parseFloat(document.getElementById('ch38_lrw').value);
    var lrl = parseFloat(document.getElementById('ch38_lrl').value);
    var beta = parseFloat(document.getElementById('ch38_beta').value);
    document.getElementById('ch38_lpw_label').textContent = lpw.toFixed(1);
    document.getElementById('ch38_lpl_label').textContent = lpl.toFixed(1);
    document.getElementById('ch38_lrw_label').textContent = lrw.toFixed(1);
    document.getElementById('ch38_lrl_label').textContent = lrl.toFixed(1);
    document.getElementById('ch38_beta_label').textContent = beta.toFixed(2);

    var rw = beta*(lpw - lrw);
    var rl = beta*(lpl - lrl);
    var diff = rw - rl;
    var sg = sigmoid(diff);
    var loss = -Math.log(sg + 1e-12);

    document.getElementById('ch38_rw').textContent = rw.toFixed(4);
    document.getElementById('ch38_rl').textContent = rl.toFixed(4);
    document.getElementById('ch38_sig').textContent = sg.toFixed(4);
    document.getElementById('ch38_loss').textContent = loss.toFixed(4);

    var plot = SL.makePlot(document.getElementById('ch38_canvas'), {l:60,r:20,t:16,b:50});
    var ctx = plot.ctx;
    var bars = [{x:0.5, v:rw, color:SL.colors.accent3, label:'winner'},
                {x:1.5, v:rl, color:SL.colors.accent4, label:'loser'}];
    var lo = Math.min(0, rw, rl), hi = Math.max(0, rw, rl);
    var pad = Math.max(0.02, (hi-lo)*0.3);
    plot.setDomain([0,2],[lo-pad, hi+pad]); plot.clear(); plot.axes('', 'implicit reward');
    plot.hline(0, SL.colors.inkSoft, true, 1);
    bars.forEach(function(b){
      ctx.fillStyle = b.color;
      var x = plot.px(b.x), y0 = plot.py(0), y1 = plot.py(b.v);
      ctx.fillRect(x-55, Math.min(y0,y1), 110, Math.abs(y1-y0));
      ctx.fillStyle = SL.colors.ink; ctx.textAlign='center'; ctx.font='12px -apple-system,sans-serif';
      ctx.fillText(b.v.toFixed(3), x, b.v>=0 ? y1-6 : y1+16);
      ctx.fillStyle = SL.colors.inkSoft;
      ctx.fillText(b.label, x, plot.py(lo-pad)+16);
    });
  }
  ['ch38_lpw','ch38_lpl','ch38_lrw','ch38_lrl','ch38_beta'].forEach(function(id){
    document.getElementById(id).addEventListener('input', update);
  });
  update();
});
</script>

## DPO vs. PPO - simplicity versus on-policy flexibility

PPO keeps three models (policy, reference, reward) and an RL loop sampling on-policy; DPO needs only policy and reference and trains supervised on fixed offline pairs, making it simpler, cheaper, and more stable - but bound to the comparisons it was given. PPO is **on-policy**: it samples fresh responses and scores them with the reward model, so it explores beyond the comparison set and can integrate signals (a compiler, Chapter 27's verifiers) a preference pair can't express. DPO is **offline and pairwise**: it learns only *relative* preference from the pairs shown and inherits their coverage gaps. DPO also has a subtle failure mode - pushing up the *relative* probability of winner over loser can drive both down while separating them, whereas PPO's reward term anchors absolute quality.

## GRPO - group-relative advantages, no value network (Chapter 7)

Another line of attack keeps the RL loop but simplifies it. **Group Relative Policy Optimization (GRPO)**, popularized for training reasoning models (e.g. DeepSeek), drops PPO's learned **value network** (the baseline $V(s)$ of Chapter 7). Instead, for each prompt it samples a *group* of several responses, scores them with the reward, and uses each response's reward *relative to the group's mean* as the advantage:

$$A_i = \frac{r_i - \text{mean}(\{r_1,\ldots,r_G\})}{\text{std}(\{r_1,\ldots,r_G\})}.$$

**Reading it piece by piece:**
- **$A_i$** - the group-relative advantage for response $i$, the weight its gradient step gets.
- **$r_i$** - the verifier's reward for that one sampled response.
- **$\text{mean}(\{r_1,\ldots,r_G\})$** - the group's average reward, acting as the free baseline to subtract.
- **$r_i - \text{mean}$** - the centered reward: positive if the sample beat its group, negative if it trailed.
- **$\text{std}(\{r_1,\ldots,r_G\})$** - the group's spread, normalizing the advantage so the gradient scale stays stable.

This is exactly the baseline idea of Chapter 7 - subtract a comparison point to reduce variance, using the group mean instead of a learned $V$ - plus a normalization that stabilizes the gradient. Dropping the value network removes an entire model and its training instability, at the cost of needing several samples per prompt. It's PPO's variance-reduction intuition, made cheaper for the large-batch language-model setting.

## Over-optimization and reward hacking still apply (Chapter 27)

None of these methods escape Chapter 27's central warning. DPO and GRPO optimize a *proxy* for human preference - comparison data or a reward model - and pushing hard against any proxy eventually exploits its specific errors: the winner's curse, Goodharting, the drift into high-scoring nonsense. DPO's $\beta$ and the reference leash play the same role as RLHF's KL penalty: they cap how far the policy can move from the sane reference, limiting how much proxy-error it can exploit. The optimizer will still find the gap between "what the preference data measured" and "what humans actually want," so the durable defenses remain better-calibrated reward signals, held-out evaluation, and regularizing toward the reference (Chapter 27's remedies).

## Worked example - by hand, larger

Take one preference pair. $y_w$ is preferred, $y_l$ rejected. We record each response's log-probability under the current policy $\pi_\theta$ and the frozen reference $\pi_{\mathrm{ref}}$, summed over tokens (Chapter 10's product-turned-sum). Suppose under the policy $\log \pi_\theta(y_w) = -2.0$ and $\log \pi_\theta(y_l) = -3.0$; under the reference $\log \pi_{\mathrm{ref}}(y_w) = -2.1$ and $\log \pi_{\mathrm{ref}}(y_l) = -2.6$; and $\beta = 0.1$.

**Step 1 - implicit reward for the winner.** The DPO implicit reward is $\hat r(y) = \beta[\log\pi_\theta(y) - \log\pi_{\mathrm{ref}}(y)]$, so

$$\hat r(y_w) = 0.1\times\big(-2.0 - (-2.1)\big) = 0.1\times(0.1) = 0.01.$$

The policy assigns $y_w$ slightly *more* probability than the reference did, giving a small positive reward.

**Step 2 - implicit reward for the loser.**

$$\hat r(y_l) = 0.1\times\big(-3.0 - (-2.6)\big) = 0.1\times(-0.4) = -0.04.$$

The policy assigns $y_l$ less probability than the reference did, giving a negative reward.

**Step 3 - the DPO logit (reward difference).** Subtract:

$$\hat r(y_w) - \hat r(y_l) = 0.01 - (-0.04) = 0.05.$$

The model only *barely* prefers the winner more than the reference did, so the logit is a small $0.05$. (This is why $\beta$ matters - it rescales this gap before the sigmoid.)

**Step 4 - sigmoid and loss.**

$$\sigma(0.05) = \frac{1}{1+e^{-0.05}} = \frac{1}{1+0.9512} = 0.5125, \qquad \mathcal{L} = -\log 0.5125 = -(-0.6683) = 0.6683 \text{ nats}.$$

DPO predicts only a $51.25\%$ chance that $y_w$ is preferred - barely better than a coin flip, so it has *almost* not learned this preference. If the model fully separated winner from loser (logit $\to \infty$, probability $\to 1$) the loss would go to $0$; $0.6683$ says the pair is far from learned.

**Gradient intuition.** The gradient of the DPO loss with respect to the logit is $-(1 - \sigma(\text{logit})) = -(1 - 0.5125) = -0.4875$: the step will *increase* the log-ratio gap, raising $\pi_\theta(y_w)$ relative to $\pi_\theta(y_l)$, and the push is large precisely because the model was only barely right - the same behavior Chapter 10 derived for cross-entropy.

**The takeaway: DPO turns one preference pair into a single number - a logistic loss on the policy-to-reference log-probability gap - and descending that loss nudges the model to make the preferred answer more likely than the rejected one, with no reward model and no reinforcement learning anywhere in the loop.**

## Check your understanding

- Why does the normalizing constant $Z(x)$ cancel when you plug the inverted reward formula into the Bradley-Terry preference probability, and why is that cancellation what makes DPO computationally feasible?
- In the DPO loss, what does subtracting the reference model's log-ratio accomplish - how does it play the same role as RLHF's KL penalty (Chapter 21)?
- Give one thing PPO can do that DPO structurally cannot, and one practical advantage of DPO over PPO.

## Exercises

### Exercise 1 - DPO loss and gradient for one pair
A preference pair has policy log-probs $\log\pi_\theta(y_w) = -1.2$, $\log\pi_\theta(y_l) = -2.0$ and reference log-probs $\log\pi_{\mathrm{ref}}(y_w) = -1.5$, $\log\pi_{\mathrm{ref}}(y_l) = -1.6$, with $\beta = 0.2$. Compute the implicit rewards for winner and loser, the logit, the sigmoid probability, and the DPO loss in nats.

<details><summary>Show solution</summary>

Implicit rewards for winner and loser:

$$\hat r(y_w) = 0.2\,[-1.2 - (-1.5)] = 0.2\times 0.3 = 0.06, \qquad \hat r(y_l) = 0.2\,[-2.0 - (-1.6)] = 0.2\times(-0.4) = -0.08.$$

The DPO logit, sigmoid, and loss:

$$\hat r(y_w) - \hat r(y_l) = 0.06 - (-0.08) = 0.14, \qquad \sigma(0.14) = \frac{1}{1+e^{-0.14}} = \frac{1}{1.8694} = 0.5349,$$

$$\mathcal{L} = -\log 0.5349 = -(-0.6257) = 0.6257 \text{ nats}.$$

The gradient w.r.t. the logit is $-(1 - 0.5349) = -0.4651$, so this step increases the log-ratio gap fairly aggressively (the model was only slightly right). As training proceeds and the logit grows, $1-\sigma \to 0$ and the gradient vanishes - the loss self-attenuates as the preference is learned.

</details>

### Exercise 2 - GRPO advantage by hand
A prompt produces a group of 4 sampled responses with rewards $r = \{2, 0, 4, 2\}$. Compute the group-relative advantage for each response using $A_i = (r_i - \text{mean})/\text{std}$ (use the population standard deviation).

<details><summary>Show solution</summary>

Mean: $\bar r = (2+0+4+2)/4 = 8/4 = 2$.

Deviations: $2-2=0$; $0-2=-2$; $4-2=2$; $2-2=0$.

Squared deviations: $0, 4, 4, 0$. Variance: $(0+4+4+0)/4 = 8/4 = 2$. Standard deviation: $\sqrt2 = 1.4142$.

Advantages: $A = \{0/1.4142,\ -2/1.4142,\ 2/1.4142,\ 0/1.4142\} = \{0,\ -1.4142,\ 1.4142,\ 0\}$.

Interpretation: the response that earned reward 4 is credited positively (+1.414) for being *better than the group average*, the reward-0 response negatively (-1.414), and the two average responses get zero - they were neither better nor worse than expected. This is exactly Chapter 7's baseline/advantage logic, with the group mean serving as the baseline and no learned value network required.

</details>

### Exercise 3 - Conceptual: when DPO's offline nature hurts
Explain why DPO can fail if the preference dataset only ever compared responses from the *initial* policy, whereas PPO (which samples fresh responses) might succeed. Frame your answer in terms of on-policy vs. offline data (Chapter 27's coverage and Chapter 22's generalization).

<details><summary>Show solution</summary>

DPO's loss only ever references the pairs it was given. If those pairs were generated by the *initial* policy, they cover only the region of response space the initial model already visits - the data is **off-policy** as soon as $\pi_\theta$ moves away. DPO never sees responses its updated policy would now produce, so it cannot correct drift or learn about behaviors absent from the fixed set - a coverage problem (Chapter 22's generalization gap, Chapter 27's offline concerns). The learned preference can fail to extrapolate beyond the training pairs.

PPO is **on-policy**: it samples fresh responses from the *current* policy, so the data distribution tracks the policy as it improves and it can explore responses the initial policy never produced. The cost is the RL machinery and its instability. DPO trades on-policy coverage for the simplicity and stability of a supervised classification loss on a fixed dataset.

</details>

## Why This Matters - In AI and In Everyday Life

**In AI / machine learning:**

- **Open-weight preference-tuned models**: Llama-style and many research models are aligned with DPO or its variants (IPO, KTO, ORPO), precisely because it turns RLHF into a stable, reward-model-free classification loss that anyone can run.
- **GRPO and reasoning-model training**: Group-relative advantages power the reward-driven training of modern reasoning models (DeepSeek-R1-style), replacing PPO's value network with a cheap group baseline (Chapter 7).
- **Alignment-pipeline design**: the closed-form $\pi^* \propto \pi_{\mathrm{ref}}e^{r/\beta}$ result explains *why* RLHF's KL leash is more than a heuristic - it's the exact optimal solution of the constrained objective (Chapter 21).
- **Mitigating reward hacking**: $\beta$ and the reference anchor in DPO, like the KL penalty in PPO, are the concrete knobs that bound over-optimization of a fallible preference proxy (Chapter 27).

**In everyday, non-AI life:**

- **Training on relative feedback**: coaching, product reviews, and hiring loops all work better when you say "prefer this over that" than when you try to assign absolute scores - the same reason pairwise data beats rating scales.
- **Constraints as regularization in decisions**: the KL-to-reference idea is the everyday principle of "improve things, but don't abandon the fundamentals you already trust" - a leash on how far optimization may wander from a known-good baseline.

## What's next

Chapter 39, **Reinforcement Learning With Verifiable Rewards - and How Reasoning Models Are Trained**, takes the next step beyond learned preference proxies entirely. When a task has a *checkable* answer - a math result you can verify, code that compiles and passes tests - you can replace the noisy, hackable reward model with ground truth, and train reasoning models with exactly the GRPO machinery and policy gradients you've now seen. It's the bridge from "predicting human preference" to "optimizing against verifiable correctness," and it's where this book's thread reaches the current frontier.