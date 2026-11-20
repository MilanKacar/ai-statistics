# Chapter 11: Confidence Intervals

## The idea this chapter builds toward

A single estimate like "$\bar x = 32.25$" (Chapter 1) hides how uncertain it is. A **confidence interval** reports a *range* of plausible values for the true parameter, built directly from the CLT (Chapter 8) and the estimator's standard error (Chapter 9), along with a precise (and commonly misunderstood) statement of how much to trust that range.

## Building the interval from pieces you already have

From Chapter 8, we know that for large enough $n$:

$$\bar X \approx N\left(\mu, \left(\frac{\sigma}{\sqrt n}\right)^2\right)$$

From Chapter 6's z-score idea, standardizing $\bar X$ gives $Z = \frac{\bar X - \mu}{\sigma/\sqrt n} \approx N(0,1)$. From the 68-95-99.7 rule, 95% of a Standard Normal distribution's mass lies between $z=-1.96$ and $z=+1.96$ (1.96 is just the precise version of "about 2" from the 95% rule of thumb). So, with 95% probability:

$$-1.96 \le \frac{\bar X - \mu}{\sigma/\sqrt n} \le 1.96$$

Rearranging this inequality to isolate $\mu$ in the middle (pure algebra — multiply through by $\sigma/\sqrt n$, then subtract $\bar X$ and flip signs):

$$\bar X - 1.96\frac{\sigma}{\sqrt n} \;\le\; \mu \;\le\; \bar X + 1.96\frac{\sigma}{\sqrt n}$$

That's the 95% confidence interval for $\mu$:

$$\text{CI} = \bar X \pm 1.96 \times SE, \qquad SE = \frac{\sigma}{\sqrt n}$$

Every confidence interval you'll ever see has this same anatomy: **point estimate ± (critical value) × (standard error)**. The "critical value" (1.96 here) just depends on how confident you want to be — 90% confidence uses 1.645, 99% confidence uses 2.576 (these come from the same Standard Normal table, just asking for different amounts of central area).

### What if you don't know $\sigma$? (the realistic case — the t-distribution)

In the formula above we assumed $\sigma$ (the true population standard deviation) was known. In real life, you almost never know $\sigma$ — you only have the sample standard deviation $s$ (Chapter 2) as an estimate of it. Substituting $s$ for $\sigma$ introduces *extra* uncertainty (you're now uncertain about two things: $\mu$ and $\sigma$, not just $\mu$), especially with small samples. William Gosset (writing as "Student," hence "Student's t-distribution") solved this: replace the Normal critical value with a **t-distribution** critical value, which has slightly heavier tails than Normal (to account for that extra uncertainty from estimating $\sigma$), controlled by a parameter called **degrees of freedom** (roughly, $n-1$ here — the same $n-1$ you first met in Chapter 2's variance formula, for closely related reasons). As $n$ grows large, the t-distribution converges to the Normal distribution (the extra uncertainty from estimating $\sigma$ becomes negligible), so for large samples, using $z=1.96$ vs. the exact t-critical-value barely matters — but for small samples (say $n<30$), it matters quite a bit, and you should always use the t-distribution unless you have a specific reason to know $\sigma$ exactly.

## What a confidence interval actually means (and the trap almost everyone falls into)

**Correct interpretation**: "If we repeated this entire sampling-and-interval-construction procedure many times, 95% of the resulting intervals would contain the true $\mu$." The 95% is a statement about the **long-run reliability of the *procedure***, not a probability statement about this one specific interval you computed.

**Common incorrect interpretation** (be careful — this is the #1 misconception in intro statistics): "there's a 95% probability that the true $\mu$ lies within *this specific* interval." This is subtly wrong under the classical (frequentist) framework, because in that framework $\mu$ is a fixed (if unknown) number, not a random variable — so for any *particular, already-computed* interval, $\mu$ either is or isn't in it; there's no probability left to speak of for that specific interval. The 95% describes the *procedure's* track record across hypothetical repeated samples, not your specific result.

(Interestingly, Bayesian statistics — Chapter 17 — has an analogous tool called a **credible interval** which *does* support the "95% probability $\mu$ is in this interval" interpretation directly, because in that framework $\mu$ is treated as a random variable with its own distribution. If you've ever found the frequentist interpretation above unsatisfying, that discomfort is legitimate, and Chapter 17 will resolve it by showing you an alternative framework built for exactly that kind of statement.)

## The backstory — Jerzy Neyman's deliberately narrow promise

**Jerzy Neyman** formalized the confidence interval in a 1937 paper, working in close (and famously combative) dialogue with Fisher and with Egon Pearson (Karl Pearson's son). Neyman was pushing statistics toward a strictly "frequentist," decision-procedure-based philosophy: rather than claim an interval captures your *degree of belief* about where $\mu$ truly is (a Bayesian-flavored claim, which Fisher himself was uneasy with for different reasons, and which the *correct* Bayesian tool — the credible interval, Chapter 17 — actually licenses), Neyman wanted something more defensible on strict long-run-frequency grounds: a *procedure* that, if you used it over and over across different problems and different data, would be right a known percentage of the time. This is precisely why the correct interpretation of a confidence interval (spelled out above) is awkwardly indirect — it's not a flaw in the explanation, it's a direct and deliberate consequence of Neyman's choice to build a guarantee about the *procedure's long-run behavior*, rather than about any single interval's probability, which he considered philosophically unjustifiable within a framework where $\mu$ is a fixed constant, not a random variable. Fisher, notably, never fully accepted Neyman's framing and promoted his own alternative ("fiducial intervals," now considered a historical dead end) — a reminder that even the giants who built this field were still arguing about what these numbers *mean*, not just how to compute them, well into the 20th century. If the interpretation feels unsatisfying to you, you're in excellent historical company.

## Worked example

Returning to our income-style sample: suppose a sample of $n=36$ customer support tickets takes an average $\bar x = 12.5$ minutes to resolve, with sample standard deviation $s=4.2$ minutes. Build a 95% CI for the true average resolution time (using the Normal approximation since $n=36$ is reasonably large):

$$SE = \frac{s}{\sqrt n} = \frac{4.2}{\sqrt{36}} = \frac{4.2}{6} = 0.7$$

$$\text{CI} = 12.5 \pm 1.96 \times 0.7 = 12.5 \pm 1.372 = [11.13,\ 13.87]$$

Interpretation: we're 95% confident (in the "long-run procedure reliability" sense above) that the true average resolution time across *all* tickets (not just these 36) is between about 11.1 and 13.9 minutes.

### What changes if you want 99% confidence instead?

Using the critical value 2.576 instead of 1.96: $\text{CI} = 12.5 \pm 2.576 \times 0.7 = 12.5 \pm 1.80 = [10.70, 14.30]$ — a **wider** interval. This is an unavoidable tradeoff: more confidence requires a wider net to be that sure you've caught the true value — you cannot get both narrower intervals *and* higher confidence from the same data; the only way to shrink the interval without losing confidence is to collect more data (shrinking $SE$ via larger $n$, straight from the CLT again).

<div class="labcard">
<div class="labkicker">Live — see what "95% confidence" really means</div>
<div class="labcontrols">
  <div class="labctrl"><label>Sample size (n) <b id="ch11_n_label">30</b></label><input type="range" id="ch11_n" min="5" max="300" value="30"></div>
  <div class="labctrl"><label>Confidence level <b id="ch11_conf_label">95%</b></label><input type="range" id="ch11_conf" min="80" max="99.5" step="0.5" value="95"></div>
  <div class="labctrl"><label>Number of experiments <b id="ch11_reps_label">100</b></label><input type="range" id="ch11_reps" min="20" max="300" value="100"></div>
</div>
<div class="labrow"><button class="labbtn" id="ch11_run">Run experiments</button></div>
<div class="labcanvas-wrap"><canvas id="ch11_canvas" width="820" height="320"></canvas></div>
<div class="labstat-grid">
  <div class="labstat"><div class="lbl">True population mean</div><div class="val" id="ch11_truemean">—</div></div>
  <div class="labstat"><div class="lbl">Intervals that captured it</div><div class="val" id="ch11_capture">—</div></div>
  <div class="labstat"><div class="lbl">Avg interval width</div><div class="val" id="ch11_width">—</div></div>
</div>
<p class="labnote">Each horizontal line is one experiment's confidence interval, drawn from a fresh random sample of a population with a known true mean. Blue intervals captured the true mean; red ones missed. Try pushing confidence to 99% with small n, then push n up to 300.</p>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var SL = window.StatsLab;
  var TRUE_MEAN = 50, TRUE_SIGMA = 15;
  function run(){
    var n = parseInt(document.getElementById('ch11_n').value);
    var confPct = parseFloat(document.getElementById('ch11_conf').value);
    var reps = parseInt(document.getElementById('ch11_reps').value);
    var conf = confPct/100;
    var captured = 0, totalWidth = 0, intervals = [];
    for(var r=0;r<reps;r++){
      var sample = []; for(var i=0;i<n;i++) sample.push(SL.randNormal(TRUE_MEAN, TRUE_SIGMA));
      var xbar = SL.mean(sample), s = SL.std(sample,1);
      var tcrit = SL.tCriticalApprox(conf, n-1);
      var se = s/Math.sqrt(n);
      var lo = xbar - tcrit*se, hi = xbar + tcrit*se;
      var ok = lo<=TRUE_MEAN && hi>=TRUE_MEAN;
      if(ok) captured++;
      totalWidth += (hi-lo);
      intervals.push([lo,hi,ok]);
    }
    document.getElementById('ch11_truemean').textContent = TRUE_MEAN.toFixed(1);
    document.getElementById('ch11_capture').textContent = captured+' / '+reps+' ('+(100*captured/reps).toFixed(1)+'%)';
    document.getElementById('ch11_width').textContent = (totalWidth/reps).toFixed(2);

    var allLo = Math.min.apply(null, intervals.map(function(i){return i[0];}).concat([TRUE_MEAN-5]));
    var allHi = Math.max.apply(null, intervals.map(function(i){return i[1];}).concat([TRUE_MEAN+5]));
    var plot = SL.makePlot(document.getElementById('ch11_canvas'), {l:50,r:20,t:10,b:30});
    plot.setDomain([allLo,allHi],[0,reps]); plot.clear(); plot.axes('value','experiment #');
    intervals.forEach(function(iv, idx){
      var color = iv[2] ? SL.colors.accent : SL.colors.accent4;
      plot.ctx.strokeStyle = color; plot.ctx.lineWidth = 1.4;
      plot.ctx.beginPath();
      plot.ctx.moveTo(plot.px(iv[0]), plot.py(idx));
      plot.ctx.lineTo(plot.px(iv[1]), plot.py(idx));
      plot.ctx.stroke();
    });
    plot.vline(TRUE_MEAN, SL.colors.ink, true);
  }
  document.getElementById('ch11_n').addEventListener('input', function(e){ document.getElementById('ch11_n_label').textContent = e.target.value; });
  document.getElementById('ch11_conf').addEventListener('input', function(e){ document.getElementById('ch11_conf_label').textContent = e.target.value+'%'; });
  document.getElementById('ch11_reps').addEventListener('input', function(e){ document.getElementById('ch11_reps_label').textContent = e.target.value; });
  document.getElementById('ch11_run').addEventListener('click', run);
  run();
});
</script>

## Why this matters for ML directly

Reporting "test accuracy = 91%" with no confidence interval is reporting a point estimate with hidden uncertainty — exactly the gap Chapter 9 flagged. A 91% accuracy on a 100-example test set has a *much* wider confidence interval than 91% on a 100,000-example test set, even though the headline number is identical — and that difference should change how much you trust a claimed improvement from, say, 91% to 92% between two models.

## More real-world examples of confidence intervals

- **Political polling**: "52% support, margin of error ±3%" is a 95% confidence interval in disguise (roughly $[49\%, 55\%]$) — built from exactly the $\pm 1.96 \times SE$ formula above, with $SE$ depending on sample size and the observed proportion.
- **Medicine/clinical trials**: a drug's effect is reported as "reduced risk by 15% (95% CI: 8%-22%)" — the width of that interval tells a reader how precisely the effect was pinned down, and whether the lower bound still represents a clinically meaningful benefit.
- **Engineering tolerances**: a component's measured strength might be reported with a confidence interval around the mean failure load, directly informing safety margins in structural design.
- **ML model benchmarking**: reporting "accuracy = 91.2% (95% CI: 89.8%-92.6%)" instead of a bare 91.2% immediately tells a reader whether a competing model's 92% result is likely a genuine improvement or indistinguishable noise given test-set size — exactly the question Chapter 12 tackles head-on.

## Figures

![100 independently constructed 95% confidence intervals from repeated samples of a known population. About 95 of them (blue) capture the true mean; a handful (red) miss it -- this is what '95% confidence' actually means.](../images/ch11_confidence_intervals_fig1.png)

*100 independently constructed 95% confidence intervals from repeated samples of a known population. About 95 of them (blue) capture the true mean; a handful (red) miss it -- this is what '95% confidence' actually means.*

## Run it yourself

`book/code/ch11_confidence_intervals.py` builds the support-ticket example above with both a z-based and t-based interval, then runs the single most convincing demonstration of what "95% confidence" really means: it repeats the sample-and-build-an-interval procedure 100 times from a *known* population and plots all 100 intervals, showing visually that almost exactly 95 of them contain the true mean and about 5 of them miss it — turning the correct (and often confusing) interpretation above into something you can literally see.

```bash
python book/code/ch11_confidence_intervals.py
```

## Why This Matters — In AI and In Everyday Life

**In AI / machine learning:**

- **Responsible model benchmarking**: reporting "accuracy = 91.2% (95% CI: 89.8%-92.6%)" instead of a bare number lets you tell whether a competitor's 92% is a real improvement or noise -- directly setting up Chapter 12.
- **Reporting uncertainty in any measured ML metric**: A/B test lift, fairness metrics, or latency improvements are all routinely reported with confidence intervals in serious ML engineering practice.

**In everyday, non-AI life:**

- **Political polling margins of error**: "52% support, margin of error +/-3%" is a 95% confidence interval in everyday news reporting, built from exactly this chapter's formula.
- **Clinical trial results**: a drug's effect is reported as "reduced risk by 15% (95% CI: 8%-22%)" so doctors and regulators know how precisely the effect was actually pinned down.

## Check your understanding

- Why does wanting 99% confidence instead of 95% force the interval to get wider, using only the ± formula?
- In your own words, explain why "95% probability that mu is in this specific interval" is not quite the right interpretation under the frequentist framework.
- Why does the t-distribution matter more for small samples than large ones?

## What's next

Chapter 12 uses this exact same machinery (point estimate, standard error, critical values) to answer a different but related question: not "what range is $\mu$ probably in," but "is there good evidence that $\mu$ differs from some specific claimed value at all" — the foundation of **hypothesis testing** and the (in)famous **p-value**.
