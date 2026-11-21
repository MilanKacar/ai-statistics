# Chapter 29: Transformers and Attention - Statistics You Can Calculate (Almost) by Hand

## The idea this chapter builds toward

Every GPT, Claude, Gemini, and Llama model - and the diffusion systems of Chapter 28 that use text conditioning - is built on one operation: **attention**. It sounds exotic, but it is a soft, data-dependent **weighted average**, which is something you already do every time you compute an expectation (Chapter 7). The whole trick of a transformer is: for each token, decide *how much to pay attention* to every other token, then take the correspondingly weighted average. This chapter strips attention down to its statistical bones - a softmax-weighted mean (Chapter 25's softmax), a variance-control scaling (Chapter 7's $\text{Var}(aX)=a^2\text{Var}(X)$), and a learned kernel smoothing (a Nadaraya-Watson estimator) - and shows you can compute one attention output entirely by hand with real numbers.

> **Intuition, in one breath:** Attention is just a weighted average where the weights are computed on the fly from how similar each item is to the query - so the model softly blends information from wherever it is most relevant.

## Attention as a weighted average - a data-dependent expectation

Chapter 7 defined the expectation of a quantity $f(X)$ as $E[f(X)] = \sum_x P(x) f(x)$: a weighted average with weights that sum to 1. Attention is exactly this, with two twists: the "weights" are computed fresh for each query from data, and the "quantity being averaged" is itself data (the values). Given a query built from one token, and a set of items (all tokens) each carrying a value, the attention output is

$$\text{out} = \sum_{j} a_j\, v_j, \qquad a_j \ge 0, \quad \sum_j a_j = 1.$$

**Reading it piece by piece:**
- **$\text{out}$** - the attention output: the blended result for the current query token, a weighted combination of everything it looked at.
- **$j$** - the index running over all tokens (all items) being attended to.
- **$v_j$** - the value of token $j$: the actual information that token contributes if attended to.
- **$a_j$** - the attention weight on token $j$: how much this query cares about that token, computed fresh from the data for each query.
- **$a_j \ge 0, \sum_j a_j = 1$** - the weights form a proper probability distribution, which is what makes the sum a true weighted mean rather than an arbitrary mixture.

It is a weighted mean, structurally identical to Chapter 7's expectation - the novelty is only in *how the weights $a_j$ are chosen*. The weights are sometimes called attention probabilities, and it is worth taking that word literally: they are a categorical distribution over the tokens (Chapter 5), so attention is a *soft, data-dependent* expectation rather than a fixed one.

## Queries, keys, and values - three views of each token

Each token is projected (via learned linear maps, Chapter 14's regression weights in disguise) into three vectors:

- **Query** $q$: "what am I looking for?" - the request from the current token.
- **Key** $k$: "what do I offer, as a matchable label?" - how a token advertises itself.
- **Value** $v$: "what information do I actually contribute if attended to?"

The natural way to score how well token $i$'s request matches token $j$'s offer is a **dot product** $q_i \cdot k_j = \sum_d q_{i,d}k_{j,d}$. This is similarity-as-inner-product, the same geometric idea behind Chapter 20's PCA and the covariance machinery of Chapter 7: vectors pointing the same way have a large positive dot product. So the raw attention scores are similarities, and we want to turn similarities into a probability distribution over tokens.

## Softmax turns scores into attention probabilities

To convert raw scores $s_j = q\cdot k_j$ into non-negative weights summing to 1, apply **softmax** (Chapter 25):

$$a_j = \frac{e^{s_j}}{\sum_{j'} e^{s_{j'}}}.$$

**Reading it piece by piece:**
- **$s_j$** - the raw similarity score between the query and token $j$ (the scaled dot product), which can be any real number.
- **$e^{s_j}$** - exponentiation of that score, which forces positivity and grows steeply with the score.
- **$\sum_{j'} e^{s_{j'}}$** - the normalizer summed over all tokens, turning the exponentiated scores into fractions that sum to one.
- **$a_j$** - the resulting attention probability for token $j$; because a big score gap makes one term dominate, softmax acts as a smooth argmax that concentrates weight on the best match.

This is the exact softmax of Chapter 25 - exponentiate to force positivity, normalize to sum to one - with the crucial statistical reading: softmax is a **smooth argmax**. For a large score gap it concentrates almost all weight on the best-matching token (like argmax, a hard maximum); for near-equal scores it spreads weight out evenly (like a uniform average). The "temperature" knob from Chapters 21/25 controls exactly this concentration: dividing scores by a temperature $T$ before softmax flattens or sharpens the attention distribution, the same knob used to control entropy at inference.

<div class="labcard">
<div class="labkicker">Live - softmax attention weights over 5 tokens</div>
<div class="labcontrols">
  <div class="labctrl"><label>Query token <b id="ch29_q_label">0</b></label><input type="range" id="ch29_q" min="0" max="4" step="1" value="0"></div>
  <div class="labctrl"><label>Temperature <b id="ch29_temp_label">1.00</b></label><input type="range" id="ch29_temp" min="0.1" max="4" step="0.05" value="1"></div>
</div>
<div class="labrow"><button class="labbtn" id="ch29_run">Run attention</button></div>
<div class="labcanvas-wrap"><canvas id="ch29_canvas" width="820" height="240"></canvas></div>
<div class="labstat-grid">
  <div class="labstat"><div class="lbl">Top weight</div><div class="val" id="ch29_top">-</div></div>
  <div class="labstat"><div class="lbl">Output[0]</div><div class="val" id="ch29_out">-</div></div>
  <div class="labstat"><div class="lbl">Entropy</div><div class="val" id="ch29_ent">-</div></div>
</div>
<p class="labnote">Bar heights are the softmax attention weights of the chosen query over the 5 tokens. Low temperature sharpens attention onto a single best-matching token; high temperature flattens it toward a uniform average.</p>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var SL = window.StatsLab;
  var Q = [[1,1,0,1],[0,1,1,0],[1,0,1,1],[0,0,1,0],[1,1,1,0]];
  var K = [[1,0,1,0],[0,1,0,1],[1,1,0,1],[0,1,1,1],[1,0,0,1]];
  var V = [[2,0,1,1],[0,2,1,0],[1,1,2,0],[0,1,0,2],[1,0,2,1]];
  function dot(a,b){ var s=0; for(var i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
  function draw(){
    var qi = parseInt(document.getElementById('ch29_q').value);
    var T = parseFloat(document.getElementById('ch29_temp').value);
    document.getElementById('ch29_q_label').textContent = qi;
    document.getElementById('ch29_temp_label').textContent = T.toFixed(2);
    var d = K[0].length;
    var scores = [], i;
    for(i=0;i<K.length;i++) scores.push(dot(Q[qi],K[i])/Math.sqrt(d)/T);
    var maxS = Math.max.apply(null,scores);
    var exps = scores.map(function(s){ return Math.exp(s-maxS); });
    var sum = exps.reduce(function(a,b){ return a+b; },0);
    if(sum <= 0) sum = 1;
    var w = exps.map(function(e){ return e/sum; });
    var ent = 0;
    w.forEach(function(p){ if(p>1e-12) ent -= p*Math.log(p); });
    var top = 0; for(i=1;i<w.length;i++) if(w[i]>w[top]) top=i;
    document.getElementById('ch29_top').textContent = 'tok '+top+' ('+(w[top]*100).toFixed(1)+'%)';
    document.getElementById('ch29_ent').textContent = ent.toFixed(3);
    var out0 = 0; for(i=0;i<w.length;i++) out0 += w[i]*V[i][0];
    document.getElementById('ch29_out').textContent = out0.toFixed(4);
    var plot = SL.makePlot(document.getElementById('ch29_canvas'));
    plot.setDomain([-0.5,4.5],[0,Math.max(0.05,Math.max.apply(null,w))*1.15]);
    plot.clear(); plot.axes('token index','attention weight');
    plot.bars(w.map(function(v,i){ return [i,v]; }), SL.colors.accent, 0.6);
    plot.vline(top, SL.colors.accent4, true);
  }
  document.getElementById('ch29_q').addEventListener('input', draw);
  document.getElementById('ch29_temp').addEventListener('input', draw);
  document.getElementById('ch29_run').addEventListener('click', draw);
  draw();
});
</script>

## Scaling by $1/\sqrt{d}$ - variance control, not decoration

The transformer divides scores by $\sqrt{d}$ (where $d$ is the key/query dimension) before softmax. This is not cosmetic. If queries and keys have $d$ independent coordinates each with mean 0 and variance 1, then the dot product $q\cdot k = \sum_{d'} q_{d'}k_{d'}$ is a sum of $d$ independent terms, each of variance 1, so (Chapter 7's variance-of-a-sum rule) $\text{Var}(q\cdot k) = d$. Dividing the score by $\sqrt{d}$ gives a quantity of variance $d \cdot (1/\sqrt{d})^2 = 1$ - exactly the $a^2\text{Var}(X)$ scaling rule from Chapter 7 applied to kill the growth. Without it, large $d$ produces huge scores, softmax saturates to near-one-hot, and gradients vanish - the same signal-explosion failure mode as bad weight initialization in Chapter 25.

## Attention is a learned kernel - the Nadaraya-Watson estimator

Now the punchline that makes attention feel inevitable. **Nadaraya-Watson kernel regression** (1964) estimates a function's value at a query $x$ by a weighted average of observed $y_j$:

$$\hat f(x) = \sum_j \frac{K\!\left(\frac{x-x_j}{h}\right)}{\sum_{j'} K\!\left(\frac{x-x_{j'}}{h}\right)}\, y_j,$$

**Reading it piece by piece:**
- **$\hat f(x)$** - the estimated function value at the query point $x$: a smooth, locally weighted prediction.
- **$x_j$ and $y_j$** - the stored inputs and their observed outputs; each observation contributes to the estimate at $x$.
- **$K$** - the kernel, a similarity function (the Gaussian bell is the classic choice) that scores how close the query is to each stored input.
- **$\frac{x-x_j}{h}$** - the gap between query and observation, rescaled by the bandwidth $h$; larger $h$ widens each kernel and smooths more.
- **the fraction** - the kernel value divided by the sum of all kernel values, normalizing similarities into weights that sum to one.
- **$\sum_j \ldots y_j$** - the weighted average of the observed outputs, where nearby inputs (large kernel) get more say.

where $K$ is a kernel (a similarity function; Chapter 6's Gaussian bell is the classic choice) and $h$ is a bandwidth. Compare to attention with a naive fixed similarity $s_j = x\cdot x_j$ and softmax weights: **they are the same equation.** Attention *is* Nadaraya-Watson kernel regression, where the kernel is learned (the $q\cdot k$ construction replaces a fixed $K$) and the softmax temperature plays the role of the bandwidth. In fact a plain Nadaraya-Watson estimator is exactly attention *without the softmax* - just normalized similarities as weights. So the transformer's core operation was hiding inside classical nonparametric regression since 1964; the transformer's contribution was making the kernel (and the whole thing) learnable and stacking it in depth.

## Positional encodings - attention is order-blind, so add order back

The weighted average $\sum_j a_j v_j$ is a **sum** - it treats its inputs as an unordered set. Permuting the tokens permutes the summands but leaves the result unchanged, so raw attention cannot tell "dog bites man" from "man bites dog." **Positional encodings** fix this by adding a position-dependent vector to each token's representation, so token identity carries a location tag. Original transformers used fixed sinusoids of varying frequency (a Fourier-style code); modern LLMs commonly use *learned* or *rotary* (RoPE) encodings. Either way, the statistical purpose is the same: inject the ordering information that a plain set-average throws away.

## Worked example - by hand, larger

Let's compute one attention output for a single query against three tokens with 2-dimensional keys and values. Suppose the query and keys/values are (we fold the $1/\sqrt{d}$, here $1/\sqrt2=0.7071$, into the scores):

Tokens with keys $k_1=(1,0)$, $k_2=(0,1)$, $k_3=(1,1)$ and values $v_1=(2,0)$, $v_2=(0,2)$, $v_3=(1,1)$. Query $q=(1,1)$.

**Raw scores (dot products).** $s_1 = q\cdot k_1 = 1\cdot1+1\cdot0 = 1$. $s_2 = 1\cdot0+1\cdot1 = 1$. $s_3 = 1\cdot1+1\cdot1 = 2$.

**Scale by $1/\sqrt2$.** $s_1 = 1\times0.7071 = 0.7071$, $s_2 = 0.7071$, $s_3 = 2\times0.7071 = 1.4142$. Now exponentiate: $e^{0.7071}=2.0281$ (twice), and $e^{1.4142}=4.1133$.

**Softmax normalization.** Sum $= 2.0281+2.0281+4.1133 = 8.1695$. So $a_1 = 2.0281/8.1695 = 0.2483$, $a_2 = 0.2483$, $a_3 = 4.1133/8.1695 = 0.5035$. Check: $0.2483+0.2483+0.5035 = 1.0001 \approx 1$. Token 3 matches best ($q\cdot k_3$ largest), so it gets roughly half the weight; tokens 1 and 2 tie.

**Weighted output.** $\text{out} = a_1 v_1 + a_2 v_2 + a_3 v_3 = 0.2483(2,0) + 0.2483(0,2) + 0.5035(1,1)$.
Component 1: $0.2483\times2 + 0.2483\times0 + 0.5035\times1 = 0.4966 + 0.5035 = 1.0001$.
Component 2: $0.2483\times0 + 0.2483\times2 + 0.5035\times1 = 0.4966 + 0.5035 = 1.0001$.
So $\text{out} \approx (1.000, 1.000)$ - neatly, the attention blended the values toward the query-aligned direction.

**The takeaway:** one attention head is literally "score each token by dot-product similarity to the query, softmax the scores into weights, take the weighted average of the values" - a learned, soft, data-dependent mean you can compute with a calculator, no magic.

## Check your understanding

- Why is attention called a "soft, data-dependent expectation," and how is that the same object as Chapter 7's $E[f(X)]=\sum_x P(x)f(x)$?
- Why divide the scores by $\sqrt{d}$? Derive the variance of an unscaled dot product of $d$ standard-Normal coordinates and explain what goes wrong without the scaling (link to Chapter 7's variance rules).
- In what sense is a transformer's attention identical to a Nadaraya-Watson kernel regression, and what makes the transformer version "learned"?

## Exercises

### Exercise 1 - Attention weights, by hand
With query $q=(2,0)$ and keys $k_1=(1,0)$, $k_2=(0,1)$, using scale $1/\sqrt2=0.7071$, compute the scaled scores and the softmax attention weights over the two tokens.

<details><summary>Show solution</summary>

Raw scores: $s_1 = q\cdot k_1 = 2\cdot1+0\cdot0 = 2$; $s_2 = q\cdot k_2 = 2\cdot0+0\cdot1 = 0$.

Scaled: $2\times0.7071 = 1.4142$; $0\times0.7071 = 0$.

Exponentials: $e^{1.4142}=4.1133$, $e^{0}=1$. Sum $=5.1133$.

Weights: $a_1 = 4.1133/5.1133 = 0.8045$, $a_2 = 1/5.1133 = 0.1955$. Check: sum $=1.000$.

Because the query points exactly along $k_1$, token 1 gets most of the weight - softmax concentrates on the best dot-product match, exactly the smooth-argmax behavior described. Raising the score gap (or lowering temperature) would push $a_1$ toward 1.

</details>

### Exercise 2 - Why scale, conceptually
Suppose $d=100$ and query/key coordinates are independent standard Normal. What is $\text{Var}(q\cdot k)$? What happens to the softmax when $d$ is large and you *forget* the $1/\sqrt{d}$ scaling, and why does that hurt training?

<details><summary>Show solution</summary>

Each term $q_{d'}k_{d'}$ has variance $1$ (product of two independent standard Normals, with $E[q_{d'}^2]=1$, $E[k_{d'}^2]=1$). Summing $d=100$ independent such terms gives $\text{Var}(q\cdot k) = 100$, so the score has standard deviation $\sqrt{100}=10$.

Unscaled, scores routinely reach $\pm30$. Feeding such large values into softmax makes it saturate: one weight races to $\approx1$ and the rest to $\approx0$, so attention becomes nearly one-hot. The gradient of softmax w.r.t. its inputs is tiny when the distribution is saturated (Chapter 25's cross-entropy gradient scales with how spread the distribution is), so gradients vanish and training stalls. Dividing by $\sqrt{d}$ renormalizes the score to variance $d\cdot(1/\sqrt{d})^2 = 1$ via Chapter 7's $a^2\text{Var}(X)$ rule, keeping softmax in a well-behaved regime regardless of head dimension.

</details>

### Exercise 3 - Positional order
Two tokens have identical keys and values but swap positions. One version of the sentence is "dog bites man," the other "man bites dog." A single attention head with no positional encoding is applied. Explain why it produces the same output for both, and how a positional encoding fixes it.

<details><summary>Show solution</summary>

Attention computes $\sum_j a_j v_j$ - a *sum*, whose value does not depend on the order of its terms. If the two tokens have identical keys and values, swapping their positions just reorders the summands, and the query (which, without positional info, also cannot see position) produces the same scores and hence the same weights. The output is therefore identical for "dog bites man" and "man bites dog," even though the meanings differ.

A positional encoding adds a position-dependent vector to each token's key/value/query representation. Now token $j$ at position 1 has a key different from token $j$ at position 2, so the dot-product scores (and the resulting weights and output) depend on order. The ordering information that the set-like weighted average discards is restored by tagging each element with its location - turning an order-blind mean into an order-aware one.

</details>

## Why This Matters - In AI and In Everyday Life

**In AI / machine learning:**

- **Every modern LLM** - GPT, Claude, Gemini, Llama - is a stack of attention layers; the softmax-weighted average computed here is the fundamental building block performed trillions of times per training run.
- **The KV cache** in inference serving stores the keys and values of already-seen tokens so each new token's attention query only needs to score against cached $(\text{key},\text{value})$ pairs - a direct engineering consequence of attention being "query against all stored keys, average the stored values," and the reason long-context inference is memory-bound.
- **Multi-head attention and long-context tricks** (grouped-query attention, sliding-window and sparse attention) are all schemes for choosing *which* tokens to include in that weighted average, trading compute against quality.
- **Temperature and top-p sampling** in text generation manipulate the same softmax-from-scores machinery this chapter uses for attention, sharing the entropy-control logic of Chapters 21 and 25.

**In everyday, non-AI life:**

- **Averaging advice from friends weighted by how relevant each is to your exact question** is a Nadaraya-Watson blend: you upweight whoever's expertise best matches your query, then combine their "values" accordingly - attention, with people as tokens.
- **A jury reaching a verdict** is a soft aggregate of opinions with more weight given to members whose reasoning best matches the questions at hand - a human weighted average that mirrors the attention operation.

## What's next

Chapter 30 turns to how a model like the transformers of this chapter is *optimized to do the right thing* - not merely to predict the next token, but to make decisions that maximize long-run reward: **Reinforcement Learning and Policy Gradients**. There you'll see Chapter 10's likelihood and Chapter 21's KL divergence reappear as the mathematics of improving a policy, connecting the generation machinery of Chapters 28 and 29 to the learning-from-feedback loop behind modern RLHF.