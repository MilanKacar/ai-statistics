# Chapter 32: Contrastive Learning and InfoNCE - Statistics That Learns Embeddings

## The idea this chapter builds toward

This chapter answers a question that has quietly become central to modern AI: how do you learn a *useful representation* of data without any labels at all? The answer that powers CLIP, SimCLR, and every retrieval system you've ever used is **contrastive learning** - pull matching things together, push mismatching things apart - and its loss function, **InfoNCE**, is not a new invention at all: it is Chapter 21's cross-entropy dressed in Chapter 25's softmax clothing, applied to similarity scores instead of class logits. Once you see that, the whole field stops looking like magic.

> **Intuition, in one breath:** An embedding is a list of numbers that places every image, word, or sentence as a point in a big coordinate space, chosen so that "similar things land near each other" - and contrastive learning is the training recipe that carves out that space by rewarding nearby friends and punishing nearby strangers, using nothing but the softmax cross-entropy you already know.

## Embeddings - turning meaning into geometry

Start with what an **embedding** actually is. An embedding is a vector $\mathbf{z} \in \mathbb{R}^d$ - just $d$ numbers - assigned to each object (an image, a word, a query), produced by a neural network. The whole point is that geometric closeness in this space should mean *semantic* closeness: two photos of dogs should sit near each other, far from a photo of a bicycle. Nothing about the raw pixels guarantees this; the vector arrangement is *learned*, and the learning is what this chapter is about.

Why learn this at all? Because once objects are points, "find similar items" becomes "find nearest points" - a fast vector lookup, not a hard reasoning problem. That single trick underlies search, recommendation, clustering, and the semantic memory inside large language models. The cost is that the network needs a *training signal* telling it how to place the points, and for unlabeled data there are no class labels to supply one. Contrastive learning manufactures that signal from the data's own structure.

When we compare two embeddings we need a similarity number. The natural one is **cosine similarity** - the angle between the vectors, which is exactly Chapter 7's correlation when the vectors are mean-centered:

$$\text{sim}(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}$$

**Reading it piece by piece:**
- **$\mathbf{a}\cdot\mathbf{b}$** - the dot product, which measures how much the two vectors point the same way before any scaling is applied.
- **$\|\mathbf{a}\|$** - the length (norm) of vector $\mathbf{a}$, the raw magnitude we want to divide out.
- **$\|\mathbf{b}\|$** - the length of vector $\mathbf{b}$, so the product $\|\mathbf{a}\|\,\|\mathbf{b}\|$ is the combined scale being cancelled.
- **$\text{sim}(\mathbf{a},\mathbf{b})$** - the resulting angle-based similarity in $[-1,+1]$, a pure direction comparison that ignores how long either vector is.

Dividing by both norms makes it scale-free, exactly as dividing covariance by $\sigma_X\sigma_Y$ made Chapter 7's correlation scale-free. Cosine similarity ranges from $-1$ (opposite) through $0$ (unrelated) to $+1$ (identical direction), and it ignores magnitude - a long vector and a short vector pointing the same way are treated as fully similar. In practice embeddings are usually L2-normalized to unit length, after which cosine similarity collapses to the plain dot product $\mathbf{a}\cdot\mathbf{b}$, and that is why the two appear interchangeably in code.

## The contrastive objective - friends close, strangers far

Now the training idea, and it is almost embarrassingly simple. Take one anchor item. Produce a **positive** pair by augmenting it - for an image, two random crops or color jitters of the *same* picture; for text, two paraphrases. Produce **negatives** by grabbing other items in the batch, which are by construction *different* objects. Then demand: the anchor's embedding should be highly similar to its positive's embedding, and dissimilar to every negative's embedding.

That is the entire objective. There is no hand-written label saying "this is a dog." The supervision is *relational* - this pair goes together, those pairs don't - and it turns out that forcing the network to solve millions of tiny "which of these is the matching one?" puzzles produces embeddings that encode genuine semantics, because the only way to reliably match crops of the same object across wildly different views is to learn what the object *is*.

This is why **augmentation matters so much**: the positive pair defines what invariances you want. If you augment with color jitter, you're telling the model "color is irrelevant, match anyway." If you augment with crops, you're saying "location within the frame doesn't matter." The augmentations literally *are* the inductive bias you're teaching.

## InfoNCE - cross-entropy over similarity scores

Time to make it a loss. For one anchor, take its similarity to its positive, $s_+$, and to $N-1$ negatives, $s_-^{(1)},\dots,s_-^{(N-1)}$. Divide everything by a **temperature** $\tau$, run the resulting scores through a **softmax** (Chapter 25), and read off the probability the model assigns to the true positive:

$$P(\text{positive} \mid \text{anchor}) = \frac{\exp(s_+/\tau)}{\exp(s_+/\tau) + \sum_{j=1}^{N-1}\exp(s_-^{(j)}/\tau)}$$

**Reading it piece by piece:**
- **$s_+$** - the similarity score between the anchor and its true positive (the augmented twin), the item we want the model to pick.
- **$s_-^{(j)}$** - the similarity score between the anchor and the $j$-th negative, one of the distractors it must reject.
- **$\tau$** - the temperature that rescales every score before exponentiation, sharpening the softmax when small and flattening it when large.
- **$\exp(\cdot)$** - the exponential that turns each score into a positive "weight", making higher similarity contribute disproportionately more mass.
- **the numerator** - the positive's weight, the model's unnormalized belief that the match is the positive.
- **the denominator** - the sum of the positive's weight and all negative weights, the total mass that normalizes the choice into a probability.
- **$P(\text{positive} \mid \text{anchor})$** - the softmax probability the model currently assigns to the correct match among all $N$ candidates.

Then take the negative log - this is **InfoNCE**, and it is *exactly* Chapter 21's cross-entropy loss applied to a categorical distribution over "which of these $N$ items is the match?":

$$\mathcal{L}_{\text{InfoNCE}} = -\log\frac{\exp(s_+/\tau)}{\sum_{k=1}^{N}\exp(s_k/\tau)}$$

**Reading it piece by piece:**
- **$s_+$** - the positive's similarity, sitting in the numerator as the "true class" logit of this $N$-way choice.
- **$s_k$** - the similarity of candidate $k$, with the denominator summing over all $N$ of them so the fraction is a proper softmax probability.
- **$\tau$** - the temperature dividing every score, setting how harshly gaps between the positive and its rivals are treated.
- **the fraction** - the model's probability that the positive is the match, exactly the softmax from the line above.
- **$-\log(\cdot)$** - the negative log that converts that probability into a penalty, large when the model is unsure and zero when it is certain.
- **$\mathcal{L}_{\text{InfoNCE}}$** - the scalar loss, which is Chapter 21's cross-entropy reduced to a single term because the true label is always the positive.

The "true class" is always index 1 (the positive), so the cross-entropy is a single negative-log term. Compare this to Chapter 21's $-\sum_x P(x)\log Q(x)$: there the true distribution $P$ was a one-hot label; here the label is "the positive is the match," and $Q$ is the softmax over similarities. The machinery is identical.

**Temperature is the concentration knob.** Dividing scores by $\tau$ before the softmax is the same move Chapter 25 used on language-model logits and Chapter 6 used to describe Normal concentration. A *small* $\tau$ sharpens the softmax - differences between $s_+$ and the negatives get amplified, so the loss brutally penalizes any negative that creeps near the positive. A *large* $\tau$ flattens it, making the loss tolerant of fuzzy score gaps. Almost every real contrastive method tunes $\tau$ carefully, because it directly controls how aggressively the space is carved.

### Reconstructing it yourself - turning it into a gradient

Push the algebra one step further, because this is where the intuition becomes a concrete force. Let $p_+ = \frac{\exp(s_+/\tau)}{\sum_k \exp(s_k/\tau)}$ be the softmax probability of the positive. Differentiating the loss with respect to a score $s_k$ gives, by the same softmax-gradient rule from Chapter 25:

$$\frac{\partial \mathcal{L}}{\partial s_k} = \frac{1}{\tau}\big(p_k - \mathbb{1}[k=\text{positive}]\big)$$

**Reading it piece by piece:**
- **$\partial \mathcal{L}/\partial s_k$** - how much the loss changes when we nudge candidate $k$'s similarity score, the force gradient descent applies to it.
- **$1/\tau$** - the temperature's inverse, which amplifies the push when $\tau$ is small and softens it when $\tau$ is large.
- **$p_k$** - the softmax probability the model currently puts on candidate $k$, i.e. how confused it is about this candidate.
- **$\mathbb{1}[k=\text{positive}]$** - an indicator that equals $1$ only for the true positive and $0$ for every negative, marking which candidate is the correct one.
- **the difference $p_k - \mathbb{1}[\cdot]$** - the "probability minus target" residual from the cross-entropy gradient: negative for the positive (so it gets pulled up) and positive for negatives (so they get pushed down).

For the positive score ($k=+$), this is $\frac{1}{\tau}(p_+ - 1)$, which is **negative** whenever $p_+<1$ - so gradient descent *increases* $s_+$, pulling the positive closer. For any negative score, it is $\frac{1}{\tau}p_k$, **positive** - so descent *decreases* $s_k$, pushing negatives away, with strength proportional to how much the model currently confuses that negative with the match ($p_k$). The model spends its effort on the *hard* negatives - the ones it wrongly thinks are the positive - which is precisely why contrastive learning is so sample-efficient. This is Chapter 21's cross-entropy gradient at work on similarity instead of logits.

<div class="labcard">
<div class="labkicker">Live - sharpen the softmax with temperature and watch the InfoNCE loss</div>
<div class="labcontrols">
  <div class="labctrl"><label>Temperature tau <b id="ch32_tau_label">0.50</b></label><input type="range" id="ch32_tau" min="0.05" max="2" step="0.05" value="0.5"></div>
  <div class="labctrl"><label>Anchor index <b id="ch32_anchor_label">0</b></label><input type="range" id="ch32_anchor" min="0" max="3" step="1" value="0"></div>
</div>
<div class="labrow">
  <button class="labbtn" id="ch32_run">Run</button>
  <button class="labbtn secondary" id="ch32_reset">Reset</button>
</div>
<div class="labcanvas-wrap"><canvas id="ch32_canvas" width="820" height="240"></canvas></div>
<div class="labstat-grid">
  <div class="labstat"><div class="lbl">InfoNCE loss (nats)</div><div class="val" id="ch32_loss">-</div></div>
  <div class="labstat"><div class="lbl">P(positive)</div><div class="val" id="ch32_ppos">-</div></div>
  <div class="labstat"><div class="lbl">Top rival prob</div><div class="val" id="ch32_rival">-</div></div>
</div>
<p class="labnote">Four fixed embeddings; item 1 is the positive for the chosen anchor and the rest are negatives. Lower tau amplifies the score gaps, concentrating probability on the positive and driving the loss toward zero; higher tau flattens the bars and keeps the loss large.</p>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var SL = window.StatsLab;
  var emb = [[1.0,0.0],[0.8,0.6],[0.1,1.0],[-0.5,0.7]];
  var labels = ['a0','p1(+)','n2','n3'];
  function dot(u,v){ var s=0; for(var i=0;i<u.length;i++) s+=u[i]*v[i]; return s; }
  function draw(){
    var tau = Math.max(1e-3, parseFloat(document.getElementById('ch32_tau').value));
    var anchor = parseInt(document.getElementById('ch32_anchor').value,10);
    if(anchor<0||anchor>=emb.length) anchor=0;
    document.getElementById('ch32_tau_label').textContent = tau.toFixed(2);
    document.getElementById('ch32_anchor_label').textContent = anchor;
    var sims = emb.map(function(e){ return dot(emb[anchor], e); });
    var exps = sims.map(function(s){ return Math.exp(s/tau); });
    var z = exps.reduce(function(a,b){return a+b;},0);
    if(!(z>0)) z = 1e-9;
    var probs = exps.map(function(e){ return e/z; });
    var posIdx = (anchor===0) ? 1 : 0;
    var ppos = probs[posIdx];
    var loss = -Math.log(Math.max(ppos,1e-12));
    document.getElementById('ch32_loss').textContent = loss.toFixed(4);
    document.getElementById('ch32_ppos').textContent = ppos.toFixed(4);
    var rival = 0, rivalIdx = -1;
    for(var i=0;i<probs.length;i++){ if(i!==posIdx && probs[i]>rival){ rival=probs[i]; rivalIdx=i; } }
    document.getElementById('ch32_rival').textContent = rival.toFixed(4);

    var plot = SL.makePlot(document.getElementById('ch32_canvas'), {l:50,r:20,t:16,b:50});
    plot.setDomain([-0.5, emb.length-0.5],[0,1.05]); plot.clear(); plot.axes('candidate', 'softmax probability');
    var ctx = plot.ctx;
    for(var k=0;k<probs.length;k++){
      var color = (k===posIdx) ? SL.colors.accent3 : SL.colors.accent4;
      ctx.fillStyle = color;
      var x = plot.px(k), y = plot.py(probs[k]), y0 = plot.py(0);
      ctx.fillRect(x-28, y, 56, y0-y);
      ctx.fillStyle = SL.colors.ink; ctx.textAlign='center'; ctx.font='12px -apple-system,sans-serif';
      ctx.fillText(probs[k].toFixed(3), x, y-6);
      ctx.fillStyle = SL.colors.inkSoft;
      ctx.fillText(labels[k], x, plot.py(0)+16);
    }
    plot.vline(anchor, SL.colors.accent, true);
  }
  document.getElementById('ch32_tau').addEventListener('input', draw);
  document.getElementById('ch32_anchor').addEventListener('input', draw);
  document.getElementById('ch32_run').addEventListener('click', draw);
  document.getElementById('ch32_reset').addEventListener('click', function(){
    document.getElementById('ch32_tau').value = '0.5';
    document.getElementById('ch32_anchor').value = '0';
    draw();
  });
  draw();
});
</script>

## Worked example - by hand, larger

Take three unit embeddings in a 2-D space: anchor $\mathbf{a}=(1,0)$, positive $\mathbf{p}=(0.8,0.6)$, and negative $\mathbf{n}=(0,1)$. All are already unit length, so cosine similarity is just the dot product. Let us compute the InfoNCE loss at temperature $\tau=0.5$ and find the gradient direction.

**Step 1 - similarities.** $s_+ = \mathbf{a}\cdot\mathbf{p} = (1)(0.8)+(0)(0.6)=0.8$. And $s_- = \mathbf{a}\cdot\mathbf{n} = (1)(0)+(0)(1)=0$. The positive scores $0.8$, the negative $0.0$.

**Step 2 - divide by temperature.** $s_+/\tau = 0.8/0.5 = 1.6$ and $s_-/\tau = 0.0/0.5 = 0.0$.

**Step 3 - exponentiate and normalize.** $\exp(1.6) = 4.953$, $\exp(0.0)=1.000$. Sum $=4.953+1.000=5.953$. Softmax probability of the positive: $p_+ = 4.953/5.953 = 0.8320$, and of the negative $p_- = 1.000/5.953 = 0.1680$.

**Step 4 - InfoNCE loss.** $\mathcal{L} = -\ln(0.8320) = 0.1839$ nats. The model already ranks the positive above the negative, so the loss is modest but nonzero - there is still room to sharpen.

**Step 5 - the gradient direction.** For the positive, $\partial\mathcal{L}/\partial s_+ = \frac{1}{0.5}(0.8320-1) = 2(-0.1680)=-0.336$. For the negative, $\partial\mathcal{L}/\partial s_- = \frac{1}{0.5}(0.1680-0) = 0.336$. Gradient descent moves in the *negative* gradient direction: it wants $s_+$ to go **up** (by $0.336$ per unit step) and $s_-$ to go **down** (by $0.336$). Concretely, nudging the positive embedding toward the anchor (increasing the angle-alignment) and the negative away will shave the loss. Note how the negative's push strength is exactly its current softmax mass $0.168$ - the confusion it causes is its punishment.

**The takeaway: contrastive learning is not a mysterious new paradigm - it is softmax cross-entropy applied to a similarity matrix, where the "correct class" is the augmented twin, and the temperature sets how hard the loss punishes adjacent strangers.**

### Why many negatives - and why batch size is everything

Look again at the denominator: it sums over all $N-1$ negatives. With only a handful of negatives, the task is trivially easy - the model can separate the positive from a few random distractors without learning anything deep. With thousands of negatives in the denominator, the model must distinguish the positive from a vast crowd, which forces it to encode fine-grained, genuinely semantic structure. This is why SimCLR's headline finding was that *larger batch sizes dramatically improve representation quality*: more in-batch negatives means a harder, richer contrastive problem per step. It is also why techniques like a **memory bank** or **momentum queue** (as in MoCo) exist - to keep thousands of negatives available even when the GPU batch is small.

### A lower bound on mutual information

Why is it called Info**NCE** - "Noise Contrastive Estimation"? Because the loss is a bound on how much the two views share. The mutual information (Chapter 21's entropy toolkit) between the anchor $x$ and its positive's context $c$ satisfies:

$$I(x;c) \ge \log N - \mathcal{L}_{\text{InfoNCE}}$$

**Reading it piece by piece:**
- **$I(x;c)$** - the mutual information between the anchor view $x$ and the positive context $c$, the shared content contrastive learning is trying to maximize.
- **$\log N$** - the entropy of choosing one of $N$ equally likely candidates, the ceiling on how much information a match can convey.
- **$\mathcal{L}_{\text{InfoNCE}}$** - the loss from above, which measures how much confusion remains about which candidate is the positive.
- **$\log N - \mathcal{L}_{\text{InfoNCE}}$** - the information actually guaranteed: start from the maximum $\log N$ and subtract the confusion the loss records.
- **$\ge$** - the inequality guaranteeing this expression never exceeds the true mutual information, making the loss a certified lower bound to be tightened by minimizing it.

Read it slowly. With $N$ candidates and a perfect classifier ($\mathcal{L}=0$), the bound gives $\log N$ - the maximum information you could possibly convey by picking one of $N$ things, which is exactly the entropy of a uniform choice over $N$ items (Chapter 21's $\log$ of the number of equally likely outcomes). If the model can always find the positive among $N$ negatives, the two views must share at least $\log N$ nats of information. If the loss is high, the views share less. So minimizing InfoNCE *maximizes a lower bound on the mutual information between the two augmented views* - the augmentations are the "channel," and contrastive learning squeezes maximum shared information through it. This is the deep reason the recipe works: it is information maximization in disguise.

## Check your understanding

- Explain, in your own words, why cosine similarity is the natural comparison for embeddings, and how it connects to the correlation coefficient from Chapter 7.
- InfoNCE is described as "cross-entropy with the positive as the true class." What exactly plays the role of the true distribution $P$ and the model distribution $Q$ from Chapter 21?
- Why does increasing the number of negatives $N$ make the learning problem harder and - paradoxically - the resulting embeddings better?

## Exercises

### Exercise 1 - InfoNCE by hand
Anchor $\mathbf{a}$, positive $\mathbf{p}$, and a single negative $\mathbf{n}$ have raw dot-product scores $s_+=2.0$ and $s_-=1.0$. Using $\tau=1.0$, compute the softmax probability of the positive and the InfoNCE loss. Then repeat with $\tau=0.25$ and explain the difference.

<details><summary>Show solution</summary>

With $\tau=1.0$: $\exp(2.0)=7.389$, $\exp(1.0)=2.718$, sum $=10.107$. So $p_+ = 7.389/10.107 = 0.7311$, and $\mathcal{L} = -\ln(0.7311) = 0.3133$ nats.

With $\tau=0.25$: $s_+/\tau = 8.0$ and $s_-/\tau = 4.0$. $\exp(8)=2980.96$, $\exp(4)=54.60$, sum $=3035.56$. So $p_+ = 2980.96/3035.56 = 0.9820$, and $\mathcal{L} = -\ln(0.9820) = 0.0182$ nats.

Lower temperature amplifies the score gap ($1.0$ becomes a gap of $4.0$ in temperature units), so the softmax concentrates almost all mass on the positive and the loss plummets. This is the concentration-control role of $\tau$: sharp temperature makes an already-correct ranking look near-perfect, and makes any error far more costly.

</details>

### Exercise 2 - The gradient on the negative
Using the same setup as Exercise 1 at $\tau=1.0$ (where $p_+=0.7311$, $p_-=0.2689$), compute $\partial\mathcal{L}/\partial s_+$ and $\partial\mathcal{L}/\partial s_-$, and state which way gradient descent moves each score.

<details><summary>Show solution</summary>

The softmax-gradient rule gives $\partial\mathcal{L}/\partial s_k = \frac{1}{\tau}(p_k-\mathbb{1}[k=+])$.

Positive: $\frac{1}{1.0}(0.7311-1) = -0.2689$. Negative: $\frac{1}{1.0}(0.2689-0)=+0.2689$.

Gradient descent steps against the gradient: it *increases* $s_+$ (by $0.2689$ per unit step) and *decreases* $s_-$ (by $0.2689$). This pulls the positive embedding closer to the anchor and pushes the negative away, with the negative's push proportional to its softmax mass $0.2689$ - the harder the model confuses it, the harder it is shoved. Notice the two magnitudes are exactly equal here because $N=2$; with many negatives each individual push is smaller but their sum drives the loss.

</details>

### Exercise 3 - Conceptual: why augmentations define the objective
A team trains a contrastive image model using only horizontal flips as augmentation, then deploys it for a task where left-right orientation matters (e.g., distinguishing the letter "b" from "d"). Predict what goes wrong and connect it to the idea that augmentations encode invariances.

<details><summary>Show solution</summary>

A horizontal flip augmentation declares "a flipped image is the same object" - it *forces* the embedding to be invariant to left-right orientation. The model therefore learns to place "b" and "d" (horizontal mirror images) at nearly the same point, destroying exactly the distinction the deployment task needs.

This is the flip side of the augmentations-are-inductive-bias claim: the model can only learn to ignore what you *tell it* to ignore, and it will aggressively ignore it. The fix is to remove the flip augmentation (or use augmentations that preserve the task-relevant structure), so the positive pair no longer asserts the invariance you don't want. The general lesson: you choose the invariances by choosing the augmentations, and choosing badly makes the embedding blind to information you later need.

</details>

## Why This Matters - In AI and In Everyday Life

**In AI / machine learning:**

- **CLIP and zero-shot classification**: CLIP trains an image encoder and a text encoder with a symmetric InfoNCE objective, so a photo's embedding lands near the embedding of its caption. The result is a shared image-text space where you can classify an unseen image category just by embedding the category name as text and finding the nearest image concept -- no retraining required.
- **Self-supervised pretraining**: SimCLR, MoCo, and BYOL learn image representations from unlabeled data with no hand-annotations at all, then fine-tune cheaply on small labeled datasets -- directly attacking the label bottleneck that Chapter 15's supervised models depend on.
- **Retrieval and vector databases**: every "semantic search" or recommendation system is nearest-neighbor lookup in an embedding space this chapter's objective built, so cosine similarity and InfoNCE are the literal mathematical substrate of modern search infrastructure.
- **Sentence embeddings and RAG**: models like Sentence-BERT use contrastive objectives to place semantically equivalent sentences near each other, which is what makes retrieval-augmented generation able to fetch the right passage for a query.

**In everyday, non-AI life:**

- **Reverse image search and "find similar"**: when you upload a photo and a service returns visually or semantically similar ones, it is ranking results by cosine similarity in a learned embedding space.
- **Music and product recommendations**: a streaming service suggests a song "because you liked this one" by placing songs as points and walking to nearby ones -- the same embedding geometry, learned from behavioral positives and negatives.

## What's next

Chapter 33 asks the question contrastive learning carefully sidesteps: matching similar things and finding structure is not the same as knowing what *causes* what. Embeddings can tell you two variables co-occur, but not whether intervening on one would change the other - and that leap, from association to intervention, is exactly what **Causal Inference and Do-Calculus** formalizes.