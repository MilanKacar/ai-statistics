# Chapter 39: Reinforcement Learning With Verifiable Rewards - and How Reasoning Models Are Trained

## The idea this chapter builds toward

Back in Chapter 27 we met the winner's curse: when you optimize hard against a *learned proxy* reward model $r(x) = q(x) + \varepsilon(x)$, you inevitably start exploiting the proxy's error $\varepsilon$, and the more you optimize (RLHF, PPO, best-of-N) the worse it gets - reward hacking, Goodhart's Law, the whole disaster. This final chapter is about escaping that trap at the root. The recipe behind o1/o3, DeepSeek-R1, and their successors is **reinforcement learning with verifiable rewards (RLVR)**, where the reward is not an estimate of correctness but *correctness itself* - a math answer you can check, code that passes tests, an exact string match. Because the reward *is* the objective, the proxy-error term $\varepsilon$ is identically zero, and the winner's curse simply cannot bite. The training loop that consumes this clean reward is the policy gradient of Chapter 30, adapted with a *group-relative* baseline (Chapter 7) called **GRPO**, and the resulting model is one whose reasoning you can scale at test time (Chapter 27).

> **Intuition, in one breath:** If you can check whether an answer is right, you can train on that check instead of on a human's guess - and a check that's simply true is a reward that nobody can game.

## Why a verifiable reward is statistically different

Chapter 27's central equation was $r(x) = q(x) + \varepsilon(x)$: the reward model's score equals true quality plus an error term the optimizer learns to exploit. Everything painful about RLHF follows from $\varepsilon(x) \neq 0$. A **verifiable reward** does something structurally different. For a math problem with answer $42$, the reward function is

$$r(x) = \begin{cases} 1 & \text{if the model's final answer is exactly } 42 \\ 0 & \text{otherwise,}\end{cases}$$

**Reading it piece by piece:**
- **$r(x)$** - the reward for a completion whose final answer is $x$, a deterministic ground-truth check.
- **$1$** - the reward when the checked answer matches exactly, the only thing that earns credit.
- **$0$** - the reward otherwise, no partial credit and no learned scorer in between.
- **the two cases** - an exact-match test (unit test, symbolic equality, regex over a boxed number) rather than a trained proxy.

a deterministic ground-truth check - a unit test, a symbolic-equality comparison, a regex over a boxed number. There is no trained scorer, no human preference model, no noise. Write it in Chapter 27's notation and you get $r(x) = q(x)$ with $\varepsilon(x) \equiv 0$.

That single change collapses the whole failure analysis. The winner's curse in Chapter 27 was the gap $E[r_{\max}] - E[q_{\text{winner}}]$ driven by selecting on positive noise; with $\varepsilon \equiv 0$, that gap is exactly zero, because the thing you select on *is* the true quality. Reward hacking has nothing to exploit: the objective is the objective. This is why Chapter 27's third "practical fix" - verifiers with ground truth instead of noisy holistic judgment - is now the *main event* rather than a patch. The catch: it only works where correctness is checkable. Code (tests), math (answers), formal logic, short exact outputs: verifiable. Open-ended essays: still needs a learned reward, and still carries the curse.

## RLVR: sampling, scoring, and the policy gradient (Chapter 30)

The training loop is exactly Chapter 30's policy gradient, framed as: the language model is the policy $\pi_\theta$, a prompt is the state, and a generated completion $y$ is a trajectory whose "return" is the verifiable reward $r(y)$. Sample $G$ completions $\{y_1, \ldots, y_G\}$ from the current policy, score each with the verifier, and take an ascent step. Plain REINFORCE would multiply each completion's log-probability gradient by its raw reward $r(y_i)$. But raw $0/1$ rewards are a terrible signal - if all $G$ samples are correct, every reward is $1$ and the gradient is nonzero but meaningless.

The fix is Chapter 7's variance machinery. Subtract a baseline. The clever part is *what* baseline: **GRPO (Group Relative Policy Optimization)** uses the **mean reward within the group of samples for that same prompt**, $\bar{r} = \frac{1}{G}\sum_{j=1}^G r(y_j)$, giving the **group-relative advantage**

$$\hat{A}_i = r(y_i) - \bar{r}.$$

**Reading it piece by piece:**
- **$\hat{A}_i$** - the group-relative advantage of completion $i$, the weight its log-probability gradient receives.
- **$r(y_i)$** - the verifier's reward for that single completion.
- **$\bar{r} = \frac{1}{G}\sum_{j=1}^G r(y_j)$** - the mean reward across the group of completions sampled for the same prompt, used as the baseline.
- **the difference** - positive when a completion beats its own group, negative when it trails, with no learned value network needed.

A completion that beats its own group gets positive advantage; one that trails it gets negative. No separate value network to train (Chapter 30's $V(s)$ is replaced by a free, in-group estimate), and it is a textbook Chapter 7 baseline: same expected gradient, much lower variance. If your verifiable reward is a small *dense* score instead of $0/1$ (e.g. tests passed), the only change is that $r$ takes more values; the estimator is identical.

<div class="labcard">
<div class="labkicker">Live - group-relative advantages and majority voting</div>
<div class="labcontrols">
  <div class="labctrl"><label>Samples in the group G <b id="ch39_g_label">8</b></label><input type="range" id="ch39_g" min="2" max="16" step="1" value="8"></div>
  <div class="labctrl"><label>Correct completions c <b id="ch39_c_label">3</b></label><input type="range" id="ch39_c" min="0" max="16" step="1" value="3"></div>
</div>
<div class="labcanvas-wrap"><canvas id="ch39_canvas" width="820" height="220"></canvas></div>
<div class="labstat-grid">
  <div class="labstat"><div class="lbl">Group mean baseline r&#772;</div><div class="val" id="ch39_mean">-</div></div>
  <div class="labstat"><div class="lbl">Advantage (correct sample)</div><div class="val" id="ch39_adv1">-</div></div>
  <div class="labstat"><div class="lbl">Advantage (wrong sample)</div><div class="val" id="ch39_adv0">-</div></div>
  <div class="labstat"><div class="lbl">pass@1 (c/G)</div><div class="val" id="ch39_pass1">-</div></div>
  <div class="labstat"><div class="lbl">Majority-vote accuracy</div><div class="val" id="ch39_maj">-</div></div>
</div>
<p class="labnote">Drag c up and down while G is fixed. A correct sample's advantage is 1 - r&#772; (positive when c &lt; G), a wrong sample's is 0 - r&#772; (negative when c &gt; 0). Note how majority voting beats pass@1 once c/G (the per-sample success probability) is above 0.5 - more samples only help the vote when the policy is already better than a coin flip.</p>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var SL = window.StatsLab;
  function logFact(n){ var s=0; for(var i=2;i<=n;i++) s+=Math.log(i); return s; }
  function binom(k,n,p){ if(k<0||k>n) return 0; var lc=logFact(n)-logFact(k)-logFact(n-k); return Math.exp(lc+k*Math.log(p+1e-12)+(n-k)*Math.log(1-p+1e-12)); }
  function update(){
    var G = parseInt(document.getElementById('ch39_g').value);
    var c = parseInt(document.getElementById('ch39_c').value);
    if(c > G) c = G;
    document.getElementById('ch39_g_label').textContent = G;
    document.getElementById('ch39_c_label').textContent = c;

    var mean = c/G;
    var p = mean;
    var pass1 = p;
    var needed = Math.floor(G/2)+1;
    var maj = 0;
    for(var k=needed;k<=G;k++) maj += binom(k,G,p);

    document.getElementById('ch39_mean').textContent = mean.toFixed(3);
    document.getElementById('ch39_adv1').textContent = (1-mean).toFixed(3);
    document.getElementById('ch39_adv0').textContent = (0-mean).toFixed(3);
    document.getElementById('ch39_pass1').textContent = pass1.toFixed(3);
    document.getElementById('ch39_maj').textContent = maj.toFixed(3);

    var plot = SL.makePlot(document.getElementById('ch39_canvas'), {l:50,r:20,t:16,b:44});
    var ctx = plot.ctx;
    var bars = [{x:0.6, v:c, color:SL.colors.accent3, label:'correct'},
                {x:1.4, v:G-c, color:SL.colors.accent4, label:'wrong'}];
    plot.setDomain([0,2],[0, G*1.2]); plot.clear(); plot.axes('', 'count');
    bars.forEach(function(b){
      ctx.fillStyle = b.color;
      var x = plot.px(b.x), y = plot.py(b.v), y0 = plot.py(0);
      ctx.fillRect(x-60, y, 120, y0-y);
      ctx.fillStyle = SL.colors.ink; ctx.textAlign='center'; ctx.font='12px -apple-system,sans-serif';
      ctx.fillText(b.v, x, y-6);
      ctx.fillStyle = SL.colors.inkSoft;
      ctx.fillText(b.label, x, plot.py(0)+16);
    });
    plot.hline(mean, SL.colors.accent, true, 2);
    ctx.fillStyle = SL.colors.accent; ctx.textAlign='left'; ctx.font='11px -apple-system,sans-serif';
    ctx.fillText('mean baseline = '+mean.toFixed(2), plot.px(0)+6, plot.py(mean)-6);
  }
  document.getElementById('ch39_g').addEventListener('input', function(){
    var G = parseInt(this.value);
    var cEl = document.getElementById('ch39_c');
    if(parseInt(cEl.value) > G) cEl.value = G;
    var cur = parseInt(cEl.value);
    document.getElementById('ch39_c_label').textContent = cur;
    update();
  });
  document.getElementById('ch39_c').addEventListener('input', update);
  update();
});
</script>

## The statistics of evaluation: pass@k, majority voting, chains of thought

Because RLVR needs a verifier, it also gives you a born-free way to *evaluate*: just check answers. Two statistics dominate. **pass@k** is the probability that at least one of $k$ independent samples is correct - the practical version of Chapter 27's best-of-N, and the reason "the model can already do it if you sample enough" is a meaningful claim: for a per-sample success probability $p$, the expected pass@k over $n$ samples is estimated as $1 - \binom{n-c}{k}/\binom{n}{k}$ ($c$ correct out of $n$). **Majority voting / self-consistency** (Chapter 27) instead generates $k$ chains of thought, extracts each final answer, and returns the mode - no verifier needed at answer time, and statistically safer because it has no proxy to Goodhart against.

Why do *longer chains of thought* help? Intuition first: a long chain is extra sequential computation - many more internal "samples" of intermediate reasoning steps - so the model effectively explores more of the space where a correct final answer lives before committing. Statistically, this is Chapter 27's test-time compute resource made *internal*: instead of drawing $N$ whole completions, one long completion draws many intermediate states, and GRPO trains the policy to make each step more likely to lead toward reward $1$.

## Process vs. outcome rewards, and training-time vs. test-time compute

An **outcome reward model (ORM)** scores only the final answer; a **process reward model (PRM)** scores each intermediate step. When the reward is truly verifiable, you get the ORM for free, so RLVR is overwhelmingly outcome-based - cheap and ungameable. A PRM (a learned, Chapter 27-style model that grades steps) can be useful where the final check is sparse and you want denser learning signal, but it reintroduces a learned proxy and hence $\varepsilon$ - the curse returns, just at the step level. As of the R1/o-series recipes, outcome-based RLVR with a clean verifier plus group-relative advantages is the dominant, best-understood recipe precisely *because* it avoids that.

Finally, Chapter 27's test-time-compute scaling and this chapter's training are two ends of one stick. RLVR trains the model so that each sample is *more likely correct* (raising $p$); self-consistency and pass@k exploit that at inference by spending more samples $N$ (Chapter 27's $\sqrt{2\ln N}$ ceiling). Better training raises the whole curve; better test-time strategy moves you along it. The two multiply.

## The remaining statistics traps

Even with a perfect verifier, statistical mischief survives - these are the open problems RLVR research keeps tripping on:

- **Deducible-format bonuses / reward shortcuts**: if the reward checks a *format* (a boxed answer, a specific XML tag) and not just correctness, the policy learns to produce the format whether or not the answer is right - a subtler Goodhart on the reward's *implementation* rather than its intent.
- **Length bias**: RLVR often rewards longer chains because more tokens mean more chances to stumble into a correct answer, so the policy can inflate length for score rather than for reasoning - a bias toward verbosity that is not genuine improvement.
- **Reward saturation**: when $p \to 1$ (or $0$), the group mean $\bar{r}$ approaches every individual $r_i$, so all advantages $\hat{A}_i \to 0$ and the gradient vanishes - learning stalls exactly when the model gets good, and you need harder prompts or a different reward to keep signal.
- **Distribution shift**: the prompts you train on define the support the policy sharpens; if evaluation uses prompts drawn from a different distribution (Chapter 9's sampling-distribution warning), measured gains can evaporate.

## Worked example - by hand, larger

Let's run one GRPO-style step on a toy verifiable-reward problem where the answer is a number, so the reward is an exact-match check. Take one prompt, sample $G = 8$ completions, and check each final answer with the verifier. Suppose **3 come out correct and 5 wrong**, so the rewards are eight values with three $1$s and five $0$s: $r = (1,1,1,0,0,0,0,0)$.

**Step 1 - the group mean baseline (Chapter 7).** Sum the rewards: $1+1+1+0+0+0+0+0 = 3$. Divide by $G=8$ to get the group baseline $\bar{r} = 3/8 = 0.375$ (which is also the empirical per-sample correctness probability $p$). This is the free, in-group baseline GRPO uses in place of a learned value network.

**Step 2 - group-relative advantages.** For each correct sample, $\hat{A} = 1 - 0.375 = +0.625$. For each wrong sample, $\hat{A} = 0 - 0.375 = -0.375$. Sanity check: the advantages must average to zero in the group, $3\times(0.625) + 5\times(-0.375) = 1.875 - 1.875 = 0$ - the sign-canceling property that makes a baseline a valid Chapter 7 variance reducer. Notice the shape: correct samples pull the policy *toward* their completions with weight $0.625$, wrong samples push *away* with weight $0.375$.

**Step 3 - the policy-gradient step (Chapter 30).** REINFORCE updates $\theta$ by $\alpha\,\hat{A}_i\,\nabla_\theta \log \pi_\theta(y_i)$ for each sample. For a correct completion with log-probability gradient $g_c$, the contribution is $+0.625\,\alpha\,g_c$ - a positive push raising that completion's probability; for a wrong completion with gradient $g_w$, it is $-0.375\,\alpha\,g_w$ - a negative push lowering it. Averaging the eight contributions, the net update raises the probability mass on the reasoning that produced $1$s and lowers it on the reasoning that produced $0$s. Iterating this across many prompts is how the policy's per-sample correctness drifts upward - nothing but Chapter 30's log-derivative gradient, weighted by a Chapter 7 baseline.

**Step 4 - evaluate: pass@1 and majority voting.** The group's correctness rate gives the per-sample estimate $\hat{p} = 0.375$, so **pass@1 is just $\hat{p} = 0.375$** - the probability a single sample is right. Now the intuition for aggregating: if each of $k$ independent samples is correct with probability $p = 0.375$ (treating $0.375$ as the true $p$), the probability that a *majority* of $k$ samples is correct rises sharply with $k$. For $k = 1$, $0.375$. For $k = 3$, the majority (at least $2$ of $3$) has probability $\binom{3}{2}p^2(1-p) + p^3 = 3(0.375^2)(0.625) + 0.375^3 = 3(0.140625)(0.625) + 0.052734 = 0.263672 + 0.052734 = 0.3164$. For $k = 9$, the binomial majority is dramatically higher - the exact Chapter 27 concentration effect, here turning a weak per-sample model into a much stronger majority-vote system without any new training.

**The takeaway: with a verifiable reward the group mean is a free, unbiased baseline, the group-relative advantages $\pm0.625$ and $-0.375$ are a pure Chapter 30 policy-gradient signal, and a single sample being right only $37.5\%$ of the time still turns into a strong majority-vote or pass@k system purely by spending test-time compute.**

## Check your understanding

- Why does a verifiable reward make the winner's-curse term $E[r_{\max}] - E[q_{\text{winner}}]$ exactly zero, in the notation of Chapter 27, while a learned reward model cannot?
- In GRPO, why is the *within-prompt* group mean $\bar{r}$ a valid baseline, and what happens to the advantages when every sample in the group gets the same reward?
- Why can a longer chain of thought improve accuracy even though it adds no new information at the final layer - and how is that the same resource Chapter 27 calls test-time compute?

## Exercises

### Exercise 1 - Group-relative advantages by hand
One prompt, $G = 10$ sampled completions, verifier marks **7 correct** and **3 wrong** (rewards $1$/$0$). Compute the group mean, the advantage of a correct sample and of a wrong sample, and verify the group's advantages sum to zero. Then say what happens to the magnitude of the correct-sample advantage if you tighten the verifier so only $4$ of $10$ are correct.

<details><summary>Show solution</summary>

Group mean: $\bar{r} = 7/10 = 0.7$. Correct-sample advantage: $1 - 0.7 = +0.3$. Wrong-sample advantage: $0 - 0.7 = -0.7$. Sum check: $7(0.3) + 3(-0.7) = 2.1 - 2.1 = 0$, confirming the baseline is valid.

Tighten the verifier to $4/10$ correct: $\bar{r} = 0.4$. Correct advantage becomes $1 - 0.4 = +0.6$ - *larger*. This is the key GRPO behavior: when correctness is rare, being right is more *surprising* relative to the group, so the advantage (and the gradient signal) is bigger. As $p$ approaches the extremes ($0$ or $1$), group advantages shrink toward zero - the reward-saturation stall.

</details>

### Exercise 2 - pass@k vs. majority vote
A model has per-sample correctness probability $p = 0.5$ on a verifiable task. Compute pass@k for $k = 1, 2, 4$ (probability at least one of $k$ independent samples is correct) and the majority-vote accuracy for $k = 3$ (assume ties broken randomly, count only strict $2$-of-$3$ as needing at least 2 correct). Compare the growth patterns.

<details><summary>Show solution</summary>

pass@k with per-sample $p=0.5$ (independent samples): probability at least one correct is $1 - (1-p)^k$.
- $k=1$: $1 - 0.5 = 0.5$.
- $k=2$: $1 - 0.5^2 = 1 - 0.25 = 0.75$.
- $k=4$: $1 - 0.5^4 = 1 - 0.0625 = 0.9375$.

Majority vote, $k=3$, needing at least 2 of 3 correct: $\binom{3}{2}0.5^2(0.5) + \binom{3}{3}0.5^3 = 3(0.125) + 0.125 = 0.375 + 0.125 = 0.5$. So a 3-of-3 majority *does not* beat $p$ at $p=0.5$ - when samples are a fair coin, the mode is just noise. This is the crucial distinction: **pass@k asks "is any sample right?" and rises fast, while majority voting asks "is the right answer also the most common?" and only helps when correct is already the per-sample mode ($p > 0.5$ for a two-answer case).** Self-consistency helps the model's *preferred* answer win; it cannot manufacture signal from a genuinely ambivalent model.

</details>

### Exercise 3 - Conceptual: why format rewards are a trap
Many RLVR setups add a small bonus for wrapping the final answer in a required format (e.g. `\boxed{}`). Explain, using Chapter 27's reward-decomposition $r(x) = q(x) + \varepsilon(x)$ and the winner's curse, why this "harmless" convenience can still cause reward hacking.

<details><summary>Show solution</summary>

The total reward is now $r(x) = r_{\text{answer}}(x) + b\cdot r_{\text{format}}(x)$. The policy maximizes the *sum*, not the answer part alone. The format term $r_{\text{format}}$ is fully controllable by the model (it's a trivially checkable string), so it is a reward component the optimizer will satisfy first and completely - producing the tag is easy and always rewarded, and the model can learn to emit the format even when the answer inside is wrong. In Chapter 27's language, the *effective* objective the optimizer sees is the intended $q(x)$ plus an extra term it can exploit with certainty; optimizing the summed proxy diverges from optimizing true correctness - a miniature Goodhart's Law where the exploit is format rather than noise.

The fix is the same as Chapter 27's: keep the exploitable component's weight small (or make it a gate that only applies *conditional on* correctness), so the reward stays dominated by the ungameable ground-truth check and the winner's curse has nothing to feed on.

</details>

## Why This Matters - In AI and In Everyday Life

**In AI / machine learning:**
- **o1/o3, DeepSeek-R1, and the reasoning-model wave** are trained with RLVR-style outcomes and group-relative policy gradients: sample many chains of thought, verify final answers, and push probability toward the reasoning that succeeds - this chapter is that recipe end to end.
- **GRPO's group baseline is a direct deployment of Chapter 7**: a prompt-local mean replaces a learned value network, cutting variance and eliminating a whole model to train, which is why it scales cleanly across large fleets of prompts.
- **pass@k, self-consistency, and test-time compute** (Chapter 27) are how these trained models are *served* - the same weights get stronger just by sampling more, and RLVR raises the per-sample $p$ that this scaling multiplies.
- **Verifier coverage is the new frontier**: as researchers push RLVR into domains without clean checkers (open-ended writing, agents, formal proofs), they fight to construct trustworthy ground-truth signals - because every step away from a true verifier reintroduces the $\varepsilon$ and the winner's curse this chapter escaped.

**In everyday, non-AI life:**
- **Practicing with feedback that's simply true**: a musician checking against the score, a coder running the test suite, a cook tasting the dish - learning fast from an *unambiguous* check rather than a vague "looks good," which is RLVR's verifier advantage in daily form.
- **The wisdom of crowds that actually works**: majority voting (self-consistency) helps when the group's most common answer is genuinely most often right, and fails when the crowd is just guessing - the same conditioning on $p > 0.5$ this chapter's statistics make precise.

## What's next

This is the end of the book - and the end of Part VII's dive into how modern AI is actually built. Where Chapter 26 gave you a map of directions to explore *depth and breadth* (causal inference, time series, experimental design, probabilistic programming), this chapter closes the loop by pointing forward *within* the frontier it described: from a verifiable reward to group-relative RL, from RL to reasoning models, from reasoning models to the test-time scaling that turns a $37.5\%$ model into a reliable one.

Look back at how far the one idea has traveled. In Chapter 1, the whole subject began with a single, almost embarrassingly modest question - what does the *average* of a handful of numbers tell you about the population it came from? That question became expectation (Chapter 7), became estimation and its error bars (Chapters 9-11), became the machinery of prediction (Chapters 14-16), and then - through likelihood, information, and the geometry of linear algebra - quietly grew into the engines underneath neural networks (Chapters 25), diffusion (Chapter 28), transformers (Chapter 29), and reinforcement learning (Chapter 30). This final chapter is that same arc arriving at its current endpoint: a sample mean became a baseline, a baseline became an advantage, an advantage became a policy gradient, and a policy gradient aimed at a *verifiable* truth is, right now, how the fastest-improving reasoning models on Earth are trained. You started by asking what a mean means; you finish knowing why the mean of a group of sampled attempts - centered, weighted, and pointed at a check that cannot lie - is what teaches a model to think.