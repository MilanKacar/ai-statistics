# Chapter 33: Causal Inference and Do-Calculus - When Correlation Is Not Enough

## The idea this chapter builds toward

Chapter 7 warned you that correlation is not causation with the ice-cream-and-drowning example, and Chapter 15 flagged Simpson's paradox without resolving it. This chapter finally builds the machinery to *do* something about it. The central object is a distinction that looks like a small notational tweak and turns out to reorganize everything: $P(Y\mid X)$ - what you see - is not $P(Y\mid do(X))$ - what happens when you *intervene*. Judea Pearl's **do-calculus** gives you rules for moving between them, and **DAGs** let you write down your causal assumptions so they can be checked instead of hidden.

> **Intuition, in one breath:** Observing that people who take a vitamin are healthier doesn't tell you whether the vitamin helps - maybe health-conscious people just take vitamins - so causal inference is the discipline of asking "what if I *forced* this?" rather than "what do the people who already have it look like?", and do-calculus is how you answer that from data you didn't have to randomize.

## Correlation vs. causation, revisited with teeth

Chapter 7's ice cream and drowning example: both rise in summer, driven by a hidden third variable, temperature. Seeing more ice cream does not make you drown. The observed association $\text{Corr}(\text{ice cream}, \text{drowning})$ is real and strong, yet the causal effect of ice cream on drowning is exactly zero. The confounder - the common cause - manufactures the association.

The mistake is subtle and pervasive: the joint distribution $P(\text{drowning}, \text{ice cream})$ genuinely does encode a statistical dependency, and a model trained to predict drowning might legitimately use ice-cream sales as a *predictive* feature. But no amount of data can turn that into "if I banned ice cream, drownings would fall." Prediction and intervention are different questions about the same numbers, and only a causal model - extra assumptions that data alone cannot supply - bridges them. That extra ingredient is the whole subject.

## Two distributions you must never conflate

Push the notation carefully, because the entire chapter lives in this gap.

$P(Y \mid X = x)$ is a **conditional**, and Chapter 3/4 already told you exactly what it is: the distribution of $Y$ among the units that *happen to have* $X=x$. It is a filter on an observed population.

$P(Y \mid do(X = x))$ is an **interventional** distribution: the distribution of $Y$ in a world where we *step in and set* $X=x$ for everyone, breaking whatever causes $X$ normally. In Pearl's notation the $do(\cdot)$ operator *deletes the arrows into $X$*. Observation just looks; intervention reaches in and rewrites the mechanism.

When do these two coincide? Precisely when the units taking $X=x$ are not otherwise different from those taking other values - i.e., when $X$ has no confounders (no common causes with $Y$). A **randomized controlled experiment** (Chapter 13's design principle) forces this by construction: the coin flip that assigns treatment is a cause of $X$ but has no other path to $Y$, so "who happened to get $X=x$" carries no information beyond $X$ itself. That single fact is why randomization is the gold standard, and why every observational method in this chapter is a patch to *simulate* it.

## Directed acyclic graphs - writing down your assumptions

A **DAG** is a graph with nodes (variables) and directed arrows (direct causal effects), with no cycles (nothing causes itself, even indirectly). Its power is that it encodes **conditional independence** assumptions in a form you can reason about and that a computer can check. Recall Chapter 3's chain rule and independence: a DAG is essentially a *factorization statement* - it says the joint distribution factorizes over each node's parents:

$$P(x_1,\dots,x_n) = \prod_i P\big(x_i \mid \text{parents}(x_i)\big)$$

**Reading it piece by piece:**
- **$P(x_1,\dots,x_n)$** - the full joint distribution over all $n$ variables, the thing the DAG claims to encode.
- **$\prod_i$** - a product over every variable, asserting the joint factorizes into one small piece per node instead of one giant table.
- **$P(x_i \mid \text{parents}(x_i))$** - the conditional probability of variable $x_i$ given only its direct parents in the graph, the local rule each arrow writes down.
- **$\text{parents}(x_i)$** - the set of variables with arrows pointing into $x_i$, the only inputs this factor is allowed to depend on.
- **the omitted variables** - every variable not in $\text{parents}(x_i)$ is asserted conditionally irrelevant, which is the independence assumption doing the real work.

Every arrow you draw asserts "this variable directly influences that one"; every arrow you *omit* asserts "there is no direct influence here." The omitted arrows do the real work - they are your assumptions, stated out loud where they can be challenged.

Three structures explain almost everything:

- **Chain**: $X \to M \to Y$. $M$ is a **mediator**. $X$ and $Y$ are dependent, but conditioning on $M$ *blocks* the path (once you know the mediator, $X$ adds nothing about $Y$).
- **Fork**: $X \leftarrow C \to Y$. $C$ is a **confounder**, a common cause. Conditioning on $C$ *blocks* the spurious path and isolates the genuine one.
- **Collider**: $X \to C \leftarrow Y$. $C$ is a common *effect*. This is the treacherous one: $X$ and $Y$ are normally independent, but conditioning on $C$ *opens* a path between them - so controlling for a collider creates a spurious association that wasn't there before. This is why "just adjust for everything you measured" is a genuinely bad heuristic.

## The backdoor criterion - how to actually compute a causal effect

The backdoor criterion tells you which variables to adjust for. A set $Z$ satisfies the **backdoor criterion** for estimating the effect of $X$ on $Y$ if (1) no member of $Z$ is a descendant of $X$, and (2) $Z$ blocks every "backdoor" path from $X$ to $Y$ - every path that enters $X$ through an arrow pointing *into* it. Intuitively, $Z$ must block all the spurious routes created by confounders while leaving the genuine $X\to Y$ routes alone.

When such a $Z$ exists, the causal effect is recovered by the **backdoor adjustment formula**:

$$P\big(Y \mid do(X=x)\big) = \sum_z P(Y \mid X=x, Z=z)\,P(Z=z)$$

**Reading it piece by piece:**
- **$P(Y \mid do(X=x))$** - the interventional distribution we want: the outcome rate if we forcibly set $X=x$ for everyone rather than watching who happens to have it.
- **$\sum_z$** - a sum over the strata of the adjustment set $Z$, blending each stratum's answer into one overall figure.
- **$P(Y \mid X=x, Z=z)$** - the observed outcome rate among units with $X=x$ *within* confounder stratum $z$, the conditional we can actually read off the data.
- **$P(Z=z)$** - the population share of stratum $z$, the weight used to average strata; note it is the natural population mix, not the mix induced by $X$.
- **the product** - each stratum's rate weighted by how common that stratum is, so the confounder is balanced as if everyone had been in every stratum.
- **the whole sum** - the standardized causal effect, which recovers $do(X=x)$ from observational data whenever $Z$ satisfies the backdoor criterion.

This is a **standardization**: within each stratum of the confounder, read off the observed conditional $Y$-rate, then average those strata using the *population* distribution of $Z$, not the distribution induced by $X$. That reweighting is the mathematical act of "as if everyone"-balancing the confounder across treatment groups. The next worked example makes this concrete.

<div class="labcard">
<div class="labkicker">Live - build Simpson's reversal and undo it with backdoor adjustment</div>
<div class="labcontrols">
  <div class="labctrl"><label>Fraction in stratum 1 <b id="ch33_frac_label">0.50</b></label><input type="range" id="ch33_frac" min="0.05" max="0.95" step="0.01" value="0.5"></div>
  <div class="labctrl"><label>Treatment advantage in stratum 1 <b id="ch33_adv1_label">+0.10</b></label><input type="range" id="ch33_adv1" min="-0.20" max="0.30" step="0.01" value="0.10"></div>
  <div class="labctrl"><label>Treatment advantage in stratum 2 <b id="ch33_adv2_label">+0.10</b></label><input type="range" id="ch33_adv2" min="-0.20" max="0.30" step="0.01" value="0.10"></div>
  <div class="labctrl"><label>Treatment share in stratum 1 <b id="ch33_treat1_label">0.20</b></label><input type="range" id="ch33_treat1" min="0.05" max="0.95" step="0.01" value="0.20"></div>
  <div class="labctrl"><label>Treatment share in stratum 2 <b id="ch33_treat2_label">0.80</b></label><input type="range" id="ch33_treat2" min="0.05" max="0.95" step="0.01" value="0.80"></div>
</div>
<div class="labrow">
  <button class="labbtn" id="ch33_run">Run</button>
  <button class="labbtn secondary" id="ch33_reset">Reset</button>
</div>
<div class="labcanvas-wrap"><canvas id="ch33_canvas" width="820" height="240"></canvas></div>
<div class="labstat-grid">
  <div class="labstat"><div class="lbl">Naive pooled diff (T-U)</div><div class="val" id="ch33_naive">-</div></div>
  <div class="labstat"><div class="lbl">Adjusted effect (backdoor)</div><div class="val" id="ch33_adj">-</div></div>
</div>
<p class="labnote">Stratum 1 has a high baseline recovery; stratum 2 a low one. If treatment is concentrated in the low-baseline stratum, the pooled difference can turn negative even though treatment helps within both strata - Simpson's reversal. The adjusted effect (weighted average of the within-stratum differences) stays positive.</p>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var SL = window.StatsLab;
  function draw(){
    var w1 = parseFloat(document.getElementById('ch33_frac').value);
    var adv1 = parseFloat(document.getElementById('ch33_adv1').value);
    var adv2 = parseFloat(document.getElementById('ch33_adv2').value);
    var t1 = parseFloat(document.getElementById('ch33_treat1').value);
    var t2 = parseFloat(document.getElementById('ch33_treat2').value);
    var w2 = 1 - w1;
    document.getElementById('ch33_frac_label').textContent = w1.toFixed(2);
    document.getElementById('ch33_adv1_label').textContent = (adv1>=0?'+':'')+adv1.toFixed(2);
    document.getElementById('ch33_adv2_label').textContent = (adv2>=0?'+':'')+adv2.toFixed(2);
    document.getElementById('ch33_treat1_label').textContent = t1.toFixed(2);
    document.getElementById('ch33_treat2_label').textContent = t2.toFixed(2);

    var base1 = 0.85, base2 = 0.20;
    var rate0_1 = base1, rate1_1 = base1 + adv1;
    var rate0_2 = base2, rate1_2 = base2 + adv2;
    rate1_1 = Math.max(0, Math.min(1, rate1_1));
    rate1_2 = Math.max(0, Math.min(1, rate1_2));

    var poolT = w1*t1*rate1_1 + w2*t2*rate1_2;
    var poolC = w1*(1-t1)*rate0_1 + w2*(1-t2)*rate0_2;
    var pooledTreatShare = w1*t1 + w2*t2;
    var pooledCtrlShare = w1*(1-t1) + w2*(1-t2);
    var rateTreat = pooledTreatShare>0 ? poolT/pooledTreatShare : 0;
    var rateCtrl = pooledCtrlShare>0 ? poolC/pooledCtrlShare : 0;
    var naive = rateTreat - rateCtrl;
    var adj = w1*adv1 + w2*adv2;

    document.getElementById('ch33_naive').textContent = (naive>=0?'+':'')+naive.toFixed(4);
    document.getElementById('ch33_adj').textContent = (adj>=0?'+':'')+adj.toFixed(4);

    var plot = SL.makePlot(document.getElementById('ch33_canvas'), {l:55,r:20,t:16,b:50});
    var vals = [rateTreat, rateCtrl, w1*adv1, w2*adv2];
    var maxV = 0.05;
    vals.forEach(function(v){ if(v>maxV) maxV=v; });
    maxV = Math.max(maxV*1.25, 0.2);
    plot.setDomain([0,4.2],[0,maxV]); plot.clear(); plot.axes('', 'recovery rate / effect');
    var ctx = plot.ctx;
    var bars = [
      {x:0.7, v:rateTreat, color:SL.colors.accent, label:'Treated (pooled)'},
      {x:1.7, v:rateCtrl, color:SL.colors.accent2, label:'Untreated (pooled)'},
      {x:3.0, v:w1*adv1, color:SL.colors.accent3, label:'Stratum 1 contrib'},
      {x:4.0, v:w2*adv2, color:SL.colors.accent3, label:'Stratum 2 contrib'}
    ];
    bars.forEach(function(b){
      ctx.fillStyle = b.color;
      var x = plot.px(b.x), y = plot.py(b.v), y0 = plot.py(0);
      ctx.fillRect(x-32, Math.min(y,y0), 64, Math.abs(y0-y));
      ctx.fillStyle = SL.colors.ink; ctx.textAlign='center'; ctx.font='12px -apple-system,sans-serif';
      ctx.fillText(b.v.toFixed(3), x, Math.min(y,y0)-6);
      ctx.fillStyle = SL.colors.inkSoft; ctx.font='11px -apple-system,sans-serif';
      ctx.fillText(b.label, x, plot.py(0)+16);
    });
    plot.vline(2.35, SL.colors.border, true);
  }
  ['ch33_frac','ch33_adv1','ch33_adv2','ch33_treat1','ch33_treat2'].forEach(function(id){
    document.getElementById(id).addEventListener('input', draw);
  });
  document.getElementById('ch33_run').addEventListener('click', draw);
  document.getElementById('ch33_reset').addEventListener('click', function(){
    document.getElementById('ch33_frac').value = '0.5';
    document.getElementById('ch33_adv1').value = '0.10';
    document.getElementById('ch33_adv2').value = '0.10';
    document.getElementById('ch33_treat1').value = '0.20';
    document.getElementById('ch33_treat2').value = '0.80';
    draw();
  });
  draw();
});
</script>

## Worked example - by hand, larger

A hospital studies whether a drug improves recovery. It is observational: doctors give the drug mostly to *older, sicker* patients, and older patients recover less on their own. We will show the naive association says the drug **hurts**, Simpson's paradox flips it, and the backdoor adjustment reveals the drug **helps**.

The confounder is age, split into "young" and "old." Here is the full table of recovery counts:

| Age group | Drug | Recovered | Total | Recovery rate |
|-----------|------|-----------|-------|---------------|
| Young | Yes | 450 | 500 | 0.90 |
| Young | No | 850 | 1000 | 0.85 |
| Old | Yes | 40 | 200 | 0.20 |
| Old | No | 5 | 50 | 0.10 |

**Step 1 - the naive (pooled) association.** Collapse over age.
Drug: $450+40 = 490$ recoveries out of $500+200 = 700$, rate $= 490/700 = 0.7000$.
No drug: $850+5 = 855$ out of $1000+50 = 1050$, rate $= 855/1050 = 0.8143$.

Naive difference: $0.7000 - 0.8143 = -0.1143$ - the drug looks like it **hurts** by about 11 percentage points. A naive analyst would kill this drug.

**Step 2 - stratified (within-age) effect.** Now slice by age. Young: drug $0.90$ vs no-drug $0.85$, a $+0.05$ benefit. Old: drug $0.20$ vs no-drug $0.10$, a $+0.10$ benefit. The drug **helps in both strata**. This is Simpson's reversal: helps everywhere, hurts in the pool, purely because the treated are overwhelmingly old (200/700 treated vs 50/1050 untreated are old), dragging the pooled treated rate into the low-baseline region.

**Step 3 - backdoor adjustment.** Apply $P(Y\mid do(X)) = \sum_z P(Y\mid X,Z)\,P(Z)$ using the **population** age weights. Total population $= 500+1000+200+50 = 1750$, so $P(\text{young}) = 1500/1750 = 0.8571$ and $P(\text{old}) = 250/1750 = 0.1429$.

Interventional recovery under drug:
$$P(Y\mid do(\text{drug})) = (0.90)(0.8571) + (0.20)(0.1429) = 0.7714 + 0.0286 = 0.8000.$$

**Reading it piece by piece:**
- **$(0.90)(0.8571)$** - the young stratum's recovery rate under the drug times the young share of the population, that stratum's contribution.
- **$(0.20)(0.1429)$** - the old stratum's recovery rate under the drug times the old share of the population.
- **$0.8571$ and $0.1429$** - the population age weights $P(\text{young})$ and $P(\text{old})$, used to rebalance the strata to the whole population.
- **$0.8000$** - the resulting interventional recovery rate if the drug were forced on everyone, the weighted blend of the two strata.

Under no drug:
$$P(Y\mid do(\text{no drug})) = (0.85)(0.8571) + (0.10)(0.1429) = 0.7286 + 0.0143 = 0.7429.$$

**Reading it piece by piece:**
- **$(0.85)(0.8571)$** - the young stratum's recovery rate without the drug, weighted by the young population share.
- **$(0.10)(0.1429)$** - the old stratum's recovery rate without the drug, weighted by the old population share.
- **$0.7429$** - the interventional recovery rate if the drug were withheld from everyone, the same standardization applied to the control arm.
- **the comparison with $0.8000$** - their difference is the causal effect, and it lands on the plus side once both arms share the same population age mix.

Adjusted causal effect: $0.8000 - 0.7429 = +0.0571$ - the drug **helps** by about 5.7 percentage points, matching the stratum-level benefits and *reversing* the naive conclusion. Notice how the standardization fixed the imbalance: it reweights the drug's outcome rates to the whole population's age mix, undoing the "treated are mostly old" distortion.

**The takeaway: the naive pooled comparison said the drug HURTS (-0.11), every within-age stratum said it HELPS, and only the backdoor-adjusted estimate (+0.057) - computed by reweighting strata to the population - recovered the truth. Simpson's paradox is not a paradox; it is what happens when you average over a confounder you failed to adjust for.**

## Randomized experiments, IVs, and selection

The cleanest fix is the one from Chapter 13: **randomize**. Assign treatment by coin flip, and the randomization arrow into $X$ has no path to $Y$ except through $X$, so there are no backdoor paths at all and $P(Y\mid do(X)) = P(Y\mid X)$ automatically. No adjustment needed - the design did the work that the arithmetic above did post hoc.

When you cannot randomize, fallbacks exist. **Instrumental variables** use a variable $I$ that affects $X$ but reaches $Y$ *only* through $X$ (no direct or confounding path) - like a lottery assignment that nudges who takes the drug without influencing recovery directly. Under strong assumptions, $I$ isolates the causal part of $X$'s variation. And **selection**: sometimes a "sample" is chosen on a variable related to both treatment and outcome (hospitalizing only severe cases), which is a collider-like bias - no amount of in-sample adjustment fixes it if the selection itself is the problem. Knowing which of these structures you face is the whole game.

## Check your understanding

- Explain the difference between $P(Y\mid X)$ and $P(Y\mid do(X))$ in terms of "who happens to have $X$" versus "who is made to have $X$, and why randomization makes them equal."
- Why does conditioning on a collider *create* an association between otherwise independent variables, whereas conditioning on a confounder *removes* one?
- In the worked example, what single feature of the data caused the pooled comparison to reverse the truth, and how did the backdoor formula correct for it?

## Exercises

### Exercise 1 - Backdoor adjustment by hand
Using the worked example's table, compute the adjusted effect if the population instead were **half young and half old** ($P(\text{young})=P(\text{old})=0.5$). Does the drug still help?

<details><summary>Show solution</summary>

Recompute the standardization with equal weights.

Under drug: $(0.90)(0.5) + (0.20)(0.5) = 0.45 + 0.10 = 0.55$.
Under no drug: $(0.85)(0.5) + (0.10)(0.5) = 0.425 + 0.05 = 0.475$.

Effect $= 0.55 - 0.475 = +0.075$. The drug helps by about 7.5 points.

This is larger than the $+0.057$ from the actual population, because the real population is mostly young (0.857), and the stratum-specific *baseline* rates are higher among the young - but the *drug's incremental benefit* is actually larger in the old stratum ($+0.10$ vs $+0.05$). So a population with more old people credits the drug with a bigger average benefit. The lesson: the adjusted effect depends on the target population's confounder mix - a genuinely causal quantity can differ across populations, which is why "the effect" always means "for this population."

</details>

### Exercise 2 - Identify the structure
For each scenario, name the structure (chain, fork, collider) and whether you should adjust for the middle variable to estimate $X$'s causal effect on $Y$: (a) Exercise $\to$ Fitness $\to$ Heart health; (b) Age $\to$ Drug $\to$ Recovery, Age $\to$ Recovery; (c) Talent $\to$ Fame $\leftarrow$ Looks (with Talent and Looks independent).

<details><summary>Show solution</summary>

(a) **Chain** ($X\to M\to Y$). Fitness is a mediator. Do **not** adjust for it - you'd block the very mechanism by which exercise helps, measuring only the direct effect and losing the indirect one. To get the total effect, leave it uncontrolled.

(b) This is a **fork/confounding** structure: Age is a common cause of both Drug (who gets treated) and Recovery. Do **adjust** for age - this is exactly the worked example, and failing to adjust is what caused Simpson's reversal.

(c) **Collider** ($X\to C\leftarrow Y$). Fame is a common effect of Talent and Looks, and the two causes are independent. Do **not** adjust for fame. Conditioning on fame (e.g., studying only famous people) *opens* a spurious path, inducing a negative association between talent and looks - among the famous, a lack of looks seems to "require" more talent. Adjusting for a collider is a classic and damaging error.

</details>

### Exercise 3 - Conceptual: why "adjust for everything" fails
A junior analyst argues: "To be safe, I'll control for every variable I measured whenever I estimate a causal effect." Give two distinct reasons this can actively make the estimate worse, referencing structures from this chapter.

<details><summary>Show solution</summary>

Two failure modes:

1. **Controlling for a mediator (chain).** If $X\to M\to Y$, adjusting for $M$ blocks the causal path and removes part (sometimes all) of the real effect. You'd wrongly conclude $X$ has little influence when in fact its influence runs *through* $M$. Over-controlling here understates or erases a genuine causal effect.

2. **Controlling for a collider.** If $X\to C\leftarrow Y$, $X$ and $Y$ may be marginally independent, but conditioning on $C$ opens a backdoor path and *manufactures* a spurious association. Adding such a variable introduces bias that wasn't present in the naive estimate.

So "adjust for everything" is not conservative - it trades one bias for others. The right move is to use a DAG to identify exactly which variables satisfy the backdoor criterion and adjust for *those*, not the whole kitchen sink. Overcontrol bias is a real, named phenomenon.

</details>

## Why This Matters - In AI and In Everyday Life

**In AI / machine learning:**

- **Counterfactual reasoning in language models**: modern models are increasingly asked "what would happen if..." questions, and answering them correctly requires the $do(\cdot)$ vs. observation distinction rather than correlational recall.
- **Fairness and bias auditing**: deciding whether a model's use of a protected attribute is discriminatory is a causal question -- is race a *cause* of the outcome or merely correlated through a confounder? -- and DAG-based adjustment is how serious audits frame it.
- **Causal feature selection and robustness**: models that latch onto spurious correlates (ice-cream-sales-style proxies) fail under distribution shift; identifying causally stable features via backdoor structure is a route to more robust predictors.
- **Off-policy evaluation and recommendation**: estimating what a system's effect *would* have been under a different policy, from logged data generated by the old policy, is exactly the confounding problem this chapter's tools address.

**In everyday, non-AI life:**

- **Reading "people who do X are healthier" headlines**: knowing that the study probably reports $P(\text{health}\mid X)$ and not $P(\text{health}\mid do(X))$ is what lets you spot a confounded claim instantly.
- **Personal decisions from observational advice**: "successful people wake up at 5am" is a correlate, not an intervention -- forcing yourself to wake early may do nothing if the real driver was something else that early risers share.

## What's next

Chapter 34 turns the DAGs of this chapter into full probabilistic machinery. Once causal structure is written as a graph, the same graph can be *run* - messages passed along its edges to compute marginals, conditionals, and likelihoods efficiently. That is **Graphical Models and Message Passing**, and it is where the structures you just drew by hand become algorithms the computer executes for you.