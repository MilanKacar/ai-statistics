# Chapter 37: Preference Data and Reward Modeling - Statistics From Human Judgments

## The idea this chapter builds toward

Every estimator so far has been fit to *numbers* - heights, counts, click-through rates. But the thing modern AI actually wants to learn is vaguer: **which of two responses a person would prefer.** This chapter shows that "which do you like better?" is still a statistical measurement problem, and that the measurements - pairwise comparisons - carry surprising amounts of usable signal even though each one is cheap, noisy, and subjective. The model that consumes them is a **reward model**, and fitting it is exactly Chapter 10's maximum likelihood applied to the **Bradley-Terry** model, which is just logistic regression (Chapter 25) wearing a ranking costume. By the end you'll see why the whole frontier of AI alignment rests on the statistical properties of a pile of human "I like this one better" clicks - and where that pile can lie to you.

> **Intuition, in one breath:** People are terrible at scoring things on an absolute 1-10 scale but remarkably consistent at saying "this one's better than that one" - so we measure preferences as head-to-head comparisons, then infer a hidden quality score for each item that best explains who won.

## Why pairwise beats absolute scoring - a measurement lesson (Chapters 12 and 13)

Ask an annotator to score a response "quality" on a 1-7 scale, and you inherit every problem Chapter 12 and Chapter 13 warned about. The scale is arbitrary (is 5 "good" or "average"?), the boundaries between 4 and 5 are fuzzy, and each person has their own internal calibration - one rater's 6 is another's 4. That's a classic **measurement-instrument** problem: you're not observing quality directly, you're observing quality through a person's private, drifting ruler. Absolute labels also invite scale compression: raters cluster in the middle and avoid extremes.

A **pairwise comparison** ("which of these two is better?") sidesteps all of it. You never need the ruler, only the *ordering*, and humans are far more reliable at comparisons than at absolute scores - the same reason Chapter 12's rank-based tests (sign test, Wilcoxon) trade a little power for robustness to the scale being wrong. The deeper statistical move is that a preference *is* a binary outcome. It's a Bernoulli trial (Chapter 5) whose probability depends on the (hidden) qualities of the two items. So each comparison is one bit of data we can feed to a likelihood - and Chapter 10 already told us exactly what to do with a pile of Bernoulli outcomes: maximize the likelihood.

This is why the entire RLHF pipeline (Chapter 30) collects comparisons, not scores: it converts a squishy subjective judgment into a clean, well-understood statistical estimation problem.

## The Bradley-Terry model (1952) - logistic regression in disguise

Give each item $i$ a latent **quality score** $r_i$ (the thing we're trying to estimate). The Bradley-Terry model - invented by Ralph Bradley and Milton Terry in 1952 to rank chess players and sports teams from match results - says the probability that item $a$ beats item $b$ is

$$P(a \text{ beats } b) = \frac{e^{r_a}}{e^{r_a} + e^{r_b}} = \sigma(r_a - r_b),$$

**Reading it piece by piece:**
- **$P(a \text{ beats } b)$** - the probability that item $a$ is preferred over item $b$, the Bernoulli outcome each comparison produces.
- **$r_a$ and $r_b$** - the latent quality scores of the two items, the hidden quantities we want to estimate.
- **$e^{r_a}/(e^{r_a}+e^{r_b})$** - $a$'s share of the total "quality weight," so a higher $r_a$ tilts the odds toward $a$.
- **$\sigma(r_a - r_b)$** - the same thing rewritten as a logistic sigmoid of the quality *difference*, showing only the gap matters.

where $\sigma(z) = \frac{1}{1+e^{-z}}$ is the logistic sigmoid. **Stop and notice what this is:** divide numerator and denominator by $e^{r_a}$ and you get $\frac{1}{1 + e^{-(r_a-r_b)}}$ - a sigmoid of a *linear* difference. That is logistic regression (Chapter 25) with the design matrix encoding "which two items were compared," and the coefficient vector being the qualities. This is exactly the connection Chapter 27 flagged when it called reward models Bradley-Terry models, and it's why Chapter 30 could slot a "learned scorer" into RLHF without inventing new theory.

To fit the $r_i$, write the likelihood over all observed comparisons and maximize it - Chapter 10's recipe verbatim. For a single comparison where $a$ actually beat $b$, the log-likelihood contribution is

$$\log P(a \text{ beat } b) = \log \sigma(r_a - r_b).$$

**Reading it piece by piece:**
- **$\log P(a \text{ beat } b)$** - the log-likelihood contribution of one observed comparison, the per-observation chunk we will sum.
- **$\sigma(r_a - r_b)$** - the model's predicted chance that $a$ wins, driven only by the quality gap.
- **$\log$** - turns the product of many probabilities into a sum, which is what makes the fit tractable.

Sum over all $N$ comparisons and maximize over the $r$'s. Because only *differences* ever appear, the qualities are identified only up to a shift - add a constant to every $r_i$ and nothing changes - so standard practice pins the scale by fixing one item's quality to zero (e.g. $r_{\text{base}} = 0$), exactly like dropping a reference level in a regression with categorical predictors (Chapter 15).

## The random-utility interpretation - where the Gaussian noise comes from (Chapter 6)

Bradley-Terry looks like a definition, but it has a *derivation*, and the derivation is the same random-utility story Chapter 6 uses for measurement error. Suppose each item $i$ has a deterministic true quality $q_i$, but when a person evaluates it under a particular context they experience a noisy value

$$U_i = q_i + \varepsilon_i, \quad \varepsilon_i \sim \text{Gumbel or Gaussian},$$

**Reading it piece by piece:**
- **$U_i$** - the noisy value a person actually experiences when judging item $i$ in a given context.
- **$q_i$** - the item's deterministic true quality, the stable part.
- **$\varepsilon_i$** - the random measurement noise around that true quality, the wobble in each human judgment.
- **Gumbel or Gaussian** - the choice of noise distribution: Gaussian gives a probit model, Gumbel gives logistic Bradley-Terry.

and the person picks whichever item feels better, $U_a > U_b$. If the noise is Gaussian with equal variance, the probability that $U_a > U_b$ is $\Phi\!\left(\frac{q_a - q_b}{\sigma\sqrt2}\right)$ - a **probit** model. If the noise is Gumbel (the extreme-value distribution Chapter 27 met while studying maxima), you get exactly the logistic sigmoid of Bradley-Terry. So Bradley-Terry is the Gumbel-noise twin of a probit model, and both are the natural answer to "probability that a noisy comparison goes one way." This matters because it tells you what the noise term *means*: each human comparison is one noisy draw from Chapter 6's distributions, and observing many of them lets you average the noise away - the CLT logic of Chapter 8 applied to preferences.

## Elo ratings - the streaming version of Bradley-Terry

You can't rerun an MLE over millions of comparisons every time a new one arrives. **Elo** (Arpad Elo, 1960s, originally for chess) is Bradley-Terry solved *online*, one comparison at a time. Keep a running rating $R_i$ for each item, convert to a 10-based scale $r_i = R_i \ln 10 / 400$, and after a comparison between $a$ and $b$, update

$$R_a \leftarrow R_a + K\,(S_a - \hat{P}_a), \qquad R_b \leftarrow R_b + K\,(S_b - \hat{P}_b),$$

**Reading it piece by piece:**
- **$R_a$ and $R_b$** - the running ratings of the two items, updated in place after each match.
- **$K$** - the step size (learning rate) controlling how far one result moves a rating.
- **$S_a$ and $S_b$** - the actual outcomes, $1$ for the winner and $0$ for the loser.
- **$\hat{P}_a$ and $\hat{P}_b$** - the predicted win probabilities before the match, from the current rating gap.
- **$S - \hat{P}$** - the prediction error (surprise); a big upset moves ratings much more than an expected result.

where $S_a \in \{0,1\}$ is the actual outcome ($1$ if $a$ won), $\hat{P}_a = \sigma(r_a-r_b)$ is the predicted win probability, and $K$ is a step size. Notice the form: it's a **gradient step on the Bradley-Terry log-likelihood** - the update is the prediction error $S - \hat P$ times a learning rate, the stochastic-gradient cousin of Chapter 10's MLE. Elo is what you'd get if you ran Bradley-Terry MLE with a single-example batch, forever, which is why it's the desktop of every online rating system: chess, tennis seeds, online games, and - increasingly - fast-moving human-preference leaderboards for language models, where new votes arrive continuously and re-fitting everything is wasteful.

<div class="labcard">
<div class="labkicker">Live - turn a head-to-head record into a rating gap</div>
<div class="labcontrols">
  <div class="labctrl"><label>Wins of A over B <b id="ch37_winsA_label">4</b></label><input type="range" id="ch37_winsA" min="0" max="20" step="1" value="4"></div>
  <div class="labctrl"><label>Wins of B over A <b id="ch37_winsB_label">1</b></label><input type="range" id="ch37_winsB" min="0" max="20" step="1" value="1"></div>
</div>
<div class="labrow">
  <button class="labbtn" id="ch37_run">Run fit</button>
  <button class="labbtn secondary" id="ch37_reset">Reset</button>
</div>
<div class="labcanvas-wrap"><canvas id="ch37_canvas" width="820" height="240"></canvas></div>
<div class="labstat-grid">
  <div class="labstat"><div class="lbl">MLE quality gap d = rA - rB</div><div class="val" id="ch37_mle">-</div></div>
  <div class="labstat"><div class="lbl">Implied P(A beats B)</div><div class="val" id="ch37_prob">-</div></div>
  <div class="labstat"><div class="lbl">Elo update (1500 vs 1500, A wins)</div><div class="val" id="ch37_elo">-</div></div>
</div>
<p class="labnote">Make the record more lopsided and watch the MLE gap d = logit(win rate) grow without bound; a 50/50 record gives d = 0 and a coin-flip win probability. The Elo row shows one online step at K = 32 from 1500 vs 1500.</p>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var SL = window.StatsLab;
  function sigmoid(z){ return 1/(1+Math.exp(-z)); }
  function update(){
    var wa = parseInt(document.getElementById('ch37_winsA').value, 10);
    var wb = parseInt(document.getElementById('ch37_winsB').value, 10);
    document.getElementById('ch37_winsA_label').textContent = wa;
    document.getElementById('ch37_winsB_label').textContent = wb;
    var n = wa + wb;
    var rate = n > 0 ? wa/n : 0.5;
    var p = rate;
    p = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
    var d = Math.log(p/(1-p));
    if(n === 0) d = 0;
    document.getElementById('ch37_mle').textContent = (n === 0 ? '0.000' : d.toFixed(3));
    document.getElementById('ch37_prob').textContent = (sigmoid(d)*100).toFixed(1)+'%';
    var K = 32;
    var eloA = 1500 + K*(1 - 0.5);
    var eloB = 1500 + K*(0 - 0.5);
    document.getElementById('ch37_elo').textContent = 'A ' + eloA.toFixed(0) + ' / B ' + eloB.toFixed(0);

    var plot = SL.makePlot(document.getElementById('ch37_canvas'));
    var maxV = Math.max(wa, wb, 1);
    plot.setDomain([0,2],[0, maxV*1.2]);
    plot.clear(); plot.axes('','wins');
    plot.bars([[0.5, wa], [1.5, wb]], SL.colors.accent, 0.5);
    var ctx = plot.ctx;
    ctx.fillStyle = SL.colors.ink; ctx.textAlign = 'center'; ctx.font = '12px -apple-system,sans-serif';
    ctx.fillText('A wins = ' + wa, plot.px(0.5), plot.py(wa)-6);
    ctx.fillText('B wins = ' + wb, plot.px(1.5), plot.py(wb)-6);
    ctx.fillStyle = SL.colors.inkSoft;
    ctx.fillText('A', plot.px(0.5), plot.py(0)+16);
    ctx.fillText('B', plot.px(1.5), plot.py(0)+16);
  }
  document.getElementById('ch37_winsA').addEventListener('input', update);
  document.getElementById('ch37_winsB').addEventListener('input', update);
  document.getElementById('ch37_run').addEventListener('click', update);
  document.getElementById('ch37_reset').addEventListener('click', function(){
    document.getElementById('ch37_winsA').value = 4;
    document.getElementById('ch37_winsB').value = 1;
    update();
  });
  update();
});
</script>

## How many comparisons do you need? Statistical power (Chapters 9 and 11)

Quality differences are estimated from a finite pile of comparisons, so each $\hat r_i$ has a standard error (Chapter 9's estimator uncertainty). The scale is set by Bernoulli variance and the number of comparisons involving the item. A useful rule of thumb: to reliably detect a quality gap corresponding to win-probability difference $\delta$, you need on the order of $N \sim 4/\delta^2$ comparisons per item pair to see it at conventional confidence levels (Chapter 11's logic: standard error falls like $1/\sqrt N$, so detecting a half-standard-deviation effect costs roughly that many samples). Ranking 100 items requires enough comparisons to make *every* pair's difference distinguishable, not just the top few - the same "compare $k$ groups, control family-wise error" worry from Chapter 13's ANOVA and multiple-comparison discussion. Under-collecting data produces confident-looking leaderboards where adjacent items are statistically indistinguishable; the fix is more comparisons, not a prettier chart.

## Noise: annotator disagreement and label noise (Chapters 2 and 22)

Real preference data is noisy in two distinct ways. First, individual annotators genuinely differ: some prefer terse answers, others prefer detail, so a "true quality" is partly a matter of the population you sampled. Second, annotator agreement is only moderate - research on heavy human-labeling tasks (e.g. the WebGPT/InstructGPT-era work, and the earlier "Learning to summarize from human feedback" paper) found inter-annotator agreement around 70% on many pairwise tasks, and roughly a third of comparison pairs get contradictory labels from different raters. That's **label noise**, the same phenomenon Chapter 22 studied in statistical learning theory as irreducible Bayes error - a floor on achievable accuracy. Chapter 2's outlier logic also applies: a few annotators clicking carelessly or with an odd agenda inject comparisons that no smooth quality model can explain. The consequence is that the best any reward model can do is match the *reducible* part of the signal; the disagreement is genuine variance, not a bug you can fit away.

## Overfitting the reward model - held-out accuracy and calibration (Chapters 16 and 25)

A reward model is itself a learned predictor, and everything Chapter 16 said about overfitting applies. It's easy to fit the *training* comparisons and then fail on new ones, so the diagnostic that matters is **held-out preference accuracy**: how often does the model correctly predict a comparison it never saw? Even more important is **calibration** (Chapter 25): does $P(a \text{ beats } b) = 0.8$ actually happen 80% of the time? A reward model that's accurate but miscalibrated, or accurate on average but overconfident near the top, will be exploited by downstream optimization - Chapter 27's reward hacking is precisely the failure that follows from a reward model being "good on held-out data" yet still a fallible proxy. The practical mandate: report held-out accuracy, check calibration curves, and never trust an accuracy number measured on the same comparisons you trained on.

## Ties and non-transitivity - when the model is simply wrong

Two biological facts of human preference break Bradley-Terry's neat assumptions. First, many comparisons end in a **tie** ("both are fine"), which the strict $S \in \{0,1\}$ model can't represent; you handle it by splitting the comparison (count it as half a win for each) or by fitting a tie-aware extension that adds a threshold around which $|q_a-q_b|$ is too small to call. Second, human preferences can be **non-transitive**: it's entirely possible that people prefer $A$ over $B$, $B$ over $C$, but $C$ over $A$ (cyclical voting, Condorcet's paradox from social-choice theory). Bradley-Terry assumes a single scalar quality per item, which *forces* transitivity - so a cyclically-preferred dataset is one the model structurally cannot represent and will fit poorly. The statistical lesson: a strong likelihood score on transitive data is evidence the scalar model is adequate; persistent non-transitivity is evidence you need richer structure (multiple quality dimensions, or context-dependent preferences).

## Worked example - by hand, larger

Suppose we've collected comparisons between just two items, A and B. The data: $A$ beat $B$ **4 times** and $B$ beat $A$ **1 time** - five comparisons total. We'll fit Bradley-Terry by hand and then run one Elo update.

**Step 1 - set up the likelihood with one parameter.** Let $d = r_A - r_B$ be the only quantity that matters (only differences are identified). From $A$'s perspective, each $A$-win has probability $\sigma(d)$ and each $B$-win has probability $\sigma(-d) = 1-\sigma(d)$. With 4 wins and 1 loss for $A$:

$$L(d) = \sigma(d)^4\,(1-\sigma(d))^1, \qquad \log L(d) = 4\log\sigma(d) + 1\log(1-\sigma(d)).$$

**Step 2 - differentiate and set to zero.** Using the clean logistic derivatives $\frac{d}{dd}\log\sigma(d) = 1-\sigma(d)$ and $\frac{d}{dd}\log(1-\sigma(d)) = -\sigma(d)$ (this is the same $\pm$ structure Chapter 30 used for policy gradients):

$$\frac{d}{dd}\log L = 4\,(1-\sigma(d)) - 1\cdot\sigma(d) = 0 \implies 4 - 4\sigma(d) - \sigma(d) = 0 \implies \sigma(d) = \frac{4}{5} = 0.8.$$

So the MLE says the *predicted* win probability of A over B is exactly the observed win rate, $4/5$ - gloriously intuitive, and exactly what Chapter 10's coin-flip MLE would have told you in disguise.

**Step 3 - recover the quality difference.** Invert the sigmoid: $d = \sigma^{-1}(0.8) = \log\frac{0.8}{1-0.8} = \log\frac{0.8}{0.2} = \log 4 = 1.3863$ (natural log). So the estimated quality gap is $\hat d = r_A - r_B = 1.3863$. If we pin the reference $r_B = 0$, then $r_A = 1.3863$. The 4-to-1 record translates to a log-odds quality lead of 1.386 - the logit of the observed win fraction, precisely the quantity logistic regression calls the linear predictor.

**Step 4 - one Elo update by hand.** Same data, but imagine it arrived one match at a time with an online rating. Start both at $R_A = R_B = 1500$, $K = 32$. Convert to the natural-log scale: $r_i = R_i \ln 10/400 = 1500 \times 2.302585/400 = 8.6347$ each, so $r_A - r_B = 0$ and the predicted win probability is $\hat P_A = \sigma(0) = 0.5$. Now the *first* comparison arrives: **A wins**, so $S_A = 1$. Update A: $R_A \leftarrow 1500 + 32(1 - 0.5) = 1500 + 16 = 1516$. Update B: $S_B = 0$, so $R_B \leftarrow 1500 + 32(0 - 0.5) = 1484$. After one game the ratings have split by 32 points, and A's win probability next time is $\sigma$ of the new gap. Repeat with the other four results (another 3 A-wins, 1 B-win) and the ratings would drift toward A being clearly stronger - the online, incremental echo of the batch MLE.

**The takeaway: a pile of pairwise human judgments collapses into a single hidden quality gap via the same logistic-MLE machinery used for ordinary classification - the MLE is just the logit of the observed win rate - and Elo is nothing more exotic than that same fit performed one comparison at a time.**

## Check your understanding

- Why can the qualities in a Bradley-Terry model only be determined up to an additive shift, and what does that force us to pin down when fitting?
- Explain why pairwise preference data is statistically preferable to absolute 1-7 quality scores, in terms of the measurement problems from Chapters 12 and 13.
- A reward model achieves 85% held-out preference accuracy but is poorly calibrated near the top of its score range. Why is that dangerous for a downstream RLHF loop (Chapter 27), even though the accuracy number looks good?

## Exercises

### Exercise 1 - Fit a two-item Bradley-Terry by hand
Item A beat item B 7 times and item B beat item A 3 times. Compute the MLE quality difference $d = r_A - r_B$ by finding the predicted win probability and inverting the sigmoid. Express the answer both as a win probability and as a log-odds.

<details><summary>Show solution</summary>

The likelihood is $L(d) = \sigma(d)^7(1-\sigma(d))^3$, so $\log L = 7\log\sigma(d) + 3\log(1-\sigma(d))$. Differentiate with the logistic derivatives:

$$\frac{d}{dd}\log L = 7(1-\sigma(d)) - 3\sigma(d) = 0 \implies 7 - 10\sigma(d) = 0 \implies \sigma(d) = \frac{7}{10} = 0.7.$$

So again the predicted win probability equals the observed win rate: $P(A \text{ beats } B) = 0.7$. Invert the sigmoid:

$$d = \log\frac{0.7}{0.3} = \log 2.333 = 0.8473.$$

So $r_A - r_B = 0.8473$: item A is favored with win probability 0.7, i.e. a log-odds quality lead of about 0.85. Note the general pattern: for $k$ wins in $n$ comparisons, $\hat d = \log\frac{k}{n-k} = \text{logit}(k/n)$ - the exact logistic-regression linear predictor for a single binary predictor.

</details>

### Exercise 2 - Elo update for a predicted vs. actual upset
Two items start at $R_A = 1800$, $R_B = 1600$, with $K = 24$. Compute the predicted win probability for A, then compute both ratings after the (a) expected result and (b) upset where B wins.

<details><summary>Show solution</summary>

Convert to the natural-log scale: the rating gap is $1800 - 1600 = 200$ points. Since $r_i = R_i\ln 10/400$, the difference is $r_A - r_B = 200 \times \ln 10/400 = 200 \times 0.005756 = 1.1513$. Predicted probability A wins:

$$\hat P_A = \sigma(1.1513) = \frac{1}{1+e^{-1.1513}} = \frac{1}{1+0.3162} = \frac{1}{1.3162} = 0.7598.$$

So A is expected to win about 76% of the time, and $\hat P_B = 0.2402$.

(a) **Expected result (A wins)**: $S_A = 1$, so $R_A \leftarrow 1800 + 24(1 - 0.7598) = 1800 + 24(0.2402) = 1800 + 5.76 = 1805.76$. And $S_B = 0$: $R_B \leftarrow 1600 + 24(0 - 0.2402) = 1600 - 5.76 = 1594.24$. A win barely moves the needle because it was already expected.

(b) **Upset (B wins)**: $S_A = 0$, so $R_A \leftarrow 1800 + 24(0 - 0.7598) = 1800 - 18.24 = 1781.76$. And $S_B = 1$: $R_B \leftarrow 1600 + 24(1 - 0.2402) = 1600 + 18.24 = 1618.24$. The upset moves the ratings much more - the update size is proportional to the *surprise* $S - \hat P$, exactly the prediction-error logic of gradient descent on the likelihood.

</details>

### Exercise 3 - Conceptual: ties and cycles
Explain (a) why a strict Bradley-Terry model cannot represent a tie, and one standard fix, and (b) why non-transitive preferences (A beats B, B beats C, C beats A) are a genuine model-misspecification problem rather than just noise.

<details><summary>Show solution</summary>

(a) Bradley-Terry assigns every comparison a probability $\sigma(r_a - r_b) \in (0,1)$ that one side wins; it never assigns positive probability mass to "neither." A tie is therefore an outcome the model gives essentially zero likelihood to (it can only be mimicked by making $r_a$ and $r_b$ nearly equal, which still predicts a near-coin-flip winner, not a tie). Standard fixes: (i) count a tie as half a win for each item (fractional outcomes), or (ii) use a tie-aware extension that declares a tie whenever $|r_a - r_b|$ falls below a learned threshold, adding a tie-zone parameter to the model.

(b) Transitivity is *forced* by the scalar structure: from $r_A > r_B$ and $r_B > r_C$ it follows automatically that $r_A > r_C$, so the model's predicted win probability for A over C is at least as high as its A-over-B probability. A genuine cycle in the data has no single scalar assignment that fits it well, so the MLE is pulled toward a compromise that mispredicts all three edges. That's misspecification - the data-generating process is richer than the model class (preferences depend on context or on multiple quality dimensions), not merely noisy. The remedy is a richer model: multiple latent quality axes (multidimensional Bradley-Terry), or context-dependent comparisons, rather than more data.

</details>

## Why This Matters - In AI and In Everyday Life

**In AI / machine learning:**

- **RLHF and instruction tuning**: InstructGPT, Claude, and essentially every aligned chat model start from a Bradley-Terry reward model (this chapter) fit to human comparisons - the reward model is the statistical heart of the entire pipeline (Chapter 30).
- **Leaderboards and evaluations**: Elo and Bradley-Terry are how chatbots, models, and players get ranked online - LMSYS Chatbot Arena and similar human-preference arenas are this chapter's streaming Elo, run on millions of fresh pairwise votes.
- **Annotator disagreement and label noise**: measuring inter-annotator agreement and modeling preference noise directly bounds the best accuracy any reward model (and any downstream policy) can achieve, exactly the irreducible-error floor of Chapter 22.
- **Reward-model overfitting and reward hacking**: held-out preference accuracy and calibration are the diagnostics that tell you whether a reward model is safe to optimize against - the first line of defense against the winner's curse and Goodharting of Chapter 27.

**In everyday, non-AI life:**

- **Any "which is better?" poll** - taste-tests, A/B ad preference panels, ranking job candidates - is a preference-measurement problem, and its noise and disagreement structure look exactly like the annotator noise described here.
- **Sports and tournament rankings**: Elo and Bradley-Terry literally came from ranking chess and sports teams, and the same "compare, don't score" logic explains why head-to-head results beat expert ratings.

## What's next

You now have a reward model: a learned scalar that predicts human preference from a pile of noisy comparisons. Chapter 38 takes that reward model and asks the RLHF question - how do you actually *optimize a policy* against it without wrecking the policy, and how can you skip the reward model entirely? The surprising answer is that the KL-regularized reward-maximization objective has a closed-form optimum, which lets you turn preference data straight into a simple binary-classification loss on the language model itself: **Direct Preference Optimization (DPO).**