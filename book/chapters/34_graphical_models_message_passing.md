# Chapter 34: Graphical Models and Message Passing - Statistics as Graphs

## The idea this chapter builds toward

A high-dimensional joint distribution is, written out naively, an exponentially large table - $2^{100}$ entries for 100 binary variables, which no machine can ever store. This chapter shows that most real joint distributions are not arbitrary tables at all: they are **products of small local factors**, one per edge in a graph, and that structure is what makes inference tractable. We will take Chapter 3's chain rule and independence, Chapter 4's evidence denominator, Chapter 6's normalizing constant, and Chapter 24's Markov chains, and assemble them into one framework: **graphical models** (Bayesian networks and Markov networks) and the **message passing** algorithms (sum-product / belief propagation, forward-backward) that do exact inference on them.

> **Intuition, in one breath:** a giant joint probability table is really just a graph of small local rules, and if you pass little messages along the edges of that graph, the correct global answer falls out without ever writing down the giant table.

## Joint distributions are exponential - unless the graph saves you

Suppose you have $n$ binary variables $X_1,\dots,X_n$. A full joint distribution has $2^n - 1$ free numbers. With $n=30$ that's already over a billion. Inference means summing out the variables you don't care about:

$$P(X_i = x) = \sum_{x_1} \cdots \sum_{x_{i-1}} \sum_{x_{i+1}} \cdots \sum_{x_n} P(X_1,\dots,X_n).$$

**Reading it piece by piece:**
- **$P(X_i = x)$** - the marginal probability we want: the chance that the single variable $X_i$ takes value $x$, summed over everything else.
- **$P(X_1,\dots,X_n)$** - the full joint distribution over all $n$ variables, the only thing we are given directly.
- **$\sum_{x_1}\cdots\sum_{x_n}$ (skipping $x_i$)** - nested sums that "average out" every other variable so only the answer for $X_i$ survives.
- **the skipped $x_i$** - $X_i$ is the variable whose value is fixed at $x$, so it is deliberately not summed over.
- **the trap** - each additional variable multiplies the work, making this an exponential $2^{n-1}$-term computation before any structure is exploited.

Done literally, that sum costs exponential time. Every result in this chapter is a way to avoid doing it literally.

## Bayesian networks: factorization via the chain rule and independence

Chapter 3 gave you the chain rule - always true, no assumptions:

$$P(X_1,\dots,X_n) = \prod_{i=1}^{n} P(X_i \mid X_1,\dots,X_{i-1}).$$

**Reading it piece by piece:**
- **$P(X_1,\dots,X_n)$** - the joint distribution over all variables, written as a product instead of enumerated as a table.
- **$\prod_{i=1}^{n}$** - a product taken one variable at a time, always exactly $n$ factors.
- **$P(X_i \mid X_1,\dots,X_{i-1})$** - the conditional for variable $i$ given every variable before it in the ordering, the chain rule applied honestly.
- **$X_1,\dots,X_{i-1}$** - the full history preceding $X_i$, which is what makes the rule unconditional and always true but potentially huge to store.
- **the caveat** - this factorization assumes nothing, so each factor can still depend on everything before it, which is why it is not yet tractable.

The problem is each factor $P(X_i \mid X_1,\dots,X_{i-1})$ can still depend on everything before it. **The Bayesian network assumption is exactly the independence assumption from Chapter 3, applied conditionally:** each variable depends only on a small set of *parents* $\text{Pa}(X_i)$, and is conditionally independent of everything else given those parents. Then the chain rule collapses to

$$P(X_1,\dots,X_n) = \prod_{i=1}^{n} P\big(X_i \mid \text{Pa}(X_i)\big).$$

**Reading it piece by piece:**
- **$P(X_1,\dots,X_n)$** - the same joint as before, now claiming a far smaller representation.
- **$\prod_{i=1}^{n}$** - still one factor per variable, but each factor only involves a handful of variables.
- **$P(X_i \mid \text{Pa}(X_i))$** - the conditional of variable $i$ given only its graph parents, the local conditional probability table (CPT).
- **$\text{Pa}(X_i)$** - the parents of $X_i$ in the DAG, the small set of variables with arrows into it; everything else has been assumed conditionally irrelevant.
- **the conditional independence assumption** - replacing the full history $X_1,\dots,X_{i-1}$ with just $\text{Pa}(X_i)$ is precisely what shrinks the table and buys tractability.

Each factor is now a tiny table (a **conditional probability table**, CPT) over the variable and its parents. The parents are drawn as arrows in a **directed acyclic graph** (DAG) - directed because the factors are conditionals, acyclic because a variable can't be its own ancestor. The graph is not decoration: **the missing edges are conditional-independence statements**, and they are what buys you tractability.

## Markov networks: undirected graphs and the partition function

Sometimes symmetry, not causation, is the right story: pixels in an image, atoms in a lattice, correlated features. Then the graph is **undirected**, and the joint is a product of **factors** $\phi_c$ over cliques $c$:

$$P(X_1,\dots,X_n) = \frac{1}{Z} \prod_{c} \phi_c(X_c), \qquad Z = \sum_{x_1,\dots,x_n} \prod_c \phi_c(X_c).$$

**Reading it piece by piece:**
- **$\prod_{c} \phi_c(X_c)$** - the unnormalized joint, a product of local factors $\phi_c$ each depending only on the variables $X_c$ in one clique $c$.
- **$\phi_c(X_c)$** - a clique potential, an arbitrary nonnegative function over that clique; unlike a CPT it need not sum to 1.
- **$X_c$** - the variables belonging to clique $c$, the scope of that one factor.
- **$1/Z$** - the normalization that rescales the product so the whole thing sums to 1, exactly Chapter 6's normalizer and Chapter 4's evidence.
- **$Z$** - the partition function, the sum of the unnormalized product over all $2^n$ configurations, the single global number that makes the model a probability distribution.
- **the cost of $Z$** - one sum over every possible joint assignment, which is why it is the computational bottleneck of undirected models.

Two old friends return here. The constant $Z$ is exactly the **normalizer** that Chapter 6 introduced for continuous densities (the $1/\sqrt{2\pi\sigma^2}$ in the Normal) and Chapter 4 called the **evidence** $P(E)$ - the quantity you divide by so the distribution sums to 1. Here it has a new name, the **partition function**, borrowed from statistical physics.

**And it is the bottleneck.** Bayesian networks could be normalized factor by factor (each CPT already sums to 1), but an undirected model's factors need not be normalized, so $Z$ is a single global sum over all $2^n$ configurations. Computing $Z$ is #P-hard in general. Almost every hard problem in graphical-model land - sampling, learning, marginal inference - traces back to this one normalizing sum.

## Message passing: exact inference on trees

A graph with no cycles - a **tree** (or chain) - lets you compute all marginals in time linear in the number of edges via **belief propagation**. The idea: each edge carries a **message** summarizing what one side of the graph believes about a shared variable, and a node's belief is the product of the messages flowing into it, times its own local factor.

For a chain, define the forward message from node $i-1$ to $i$ recursively:

$$m_{i-1 \to i}(x_i) = \sum_{x_{i-1}} \phi(x_{i-1}, x_i)\, \phi(x_{i-1})\, m_{i-2 \to i-1}(x_{i-1}).$$

**Reading it piece by piece:**
- **$m_{i-1 \to i}(x_i)$** - the message node $i-1$ sends to node $i$, a function of $x_i$ summarizing everything the left side of the chain believes about it.
- **$\sum_{x_{i-1}}$** - a sum over the sender's own values, which marginalizes out $x_{i-1}$ one variable at a time instead of over the whole joint.
- **$\phi(x_{i-1}, x_i)$** - the pairwise factor on the edge, the local coupling between the sender and receiver.
- **$\phi(x_{i-1})$** - the sender's own local factor, its private evidence.
- **$m_{i-2 \to i-1}(x_{i-1})$** - the message the sender already received from further left, so information flows cumulatively down the chain.
- **the whole recursion** - each message is built from the previous message plus local factors, replacing an exponential sum with a linear chain of small sums.

Then the marginal is just the product of incoming messages, normalized:

$$P(x_i) \propto \phi(x_i)\, m_{i-1\to i}(x_i)\, m_{i+1\to i}(x_i).$$

**Reading it piece by piece:**
- **$P(x_i)$** - the marginal belief at node $i$, what we ultimately want, up to the constant of proportionality.
- **$\phi(x_i)$** - the node's own local factor, its private evidence independent of its neighbors.
- **$m_{i-1\to i}(x_i)$** - the message arriving from the left neighbor, summarizing everything on that side.
- **$m_{i+1\to i}(x_i)$** - the message arriving from the right neighbor, summarizing everything on that side.
- **the product** - combining the node's own factor with all incoming messages, which is how two independently-computed summaries merge into one exact belief.
- **$\propto$** - a reminder that the product must be renormalized to sum to 1, since messages carry only relative weights.

This is the **sum-product algorithm**. It replaces an exponential sum over all variables with a sequence of local sums over one variable at a time. The factor-graph formulation (with explicit factor nodes) makes the same recursion work on any tree and is the form implementations actually use.

<div class="labcard">
<div class="labkicker">Live - brute-force sum vs. message passing, side by side</div>
<div class="labcontrols">
  <div class="labctrl"><label>P(B=1 | A=1) <b id="ch34_pba_label">0.80</b></label><input type="range" id="ch34_pba" min="0.05" max="0.95" step="0.05" value="0.80"></div>
  <div class="labctrl"><label>P(C=1 | B=1) <b id="ch34_pcb_label">0.70</b></label><input type="range" id="ch34_pcb" min="0.05" max="0.95" step="0.05" value="0.70"></div>
  <div class="labctrl"><label>P(A=1) <b id="ch34_pa_label">0.50</b></label><input type="range" id="ch34_pa" min="0.05" max="0.95" step="0.05" value="0.50"></div>
</div>
<div class="labcanvas-wrap"><canvas id="ch34_canvas" width="820" height="200"></canvas></div>
<div class="labstat-grid">
  <div class="labstat"><div class="lbl">P(B=1) brute force</div><div class="val" id="ch34_brute">-</div></div>
  <div class="labstat"><div class="lbl">P(B=1) messages</div><div class="val" id="ch34_msg">-</div></div>
  <div class="labstat"><div class="lbl">Match?</div><div class="val" id="ch34_match">-</div></div>
</div>
<p class="labnote">The chain is A &rarr; B &rarr; C with P(B=1|A=0) = P(C=1|B=0) held at 0.2. Brute force sums the joint over every (A, C) pair; message passing multiplies the two incoming messages at B. They always agree - that agreement is the theorem, demonstrated numerically.</p>
</div>
<script>
document.addEventListener('DOMContentLoaded', function(){
  var SL = window.StatsLab;
  function update(){
    var pa1 = parseFloat(document.getElementById('ch34_pa').value);
    var pba1 = parseFloat(document.getElementById('ch34_pba').value);
    var pcb1 = parseFloat(document.getElementById('ch34_pcb').value);
    document.getElementById('ch34_pa_label').textContent = pa1.toFixed(2);
    document.getElementById('ch34_pba_label').textContent = pba1.toFixed(2);
    document.getElementById('ch34_pcb_label').textContent = pcb1.toFixed(2);
    var C_GIVEN_0 = 0.2, B_GIVEN_A0 = 0.2;

    var pA = [1-pa1, pa1];
    var pBgivenA = [[1-B_GIVEN_A0, B_GIVEN_A0],[1-pba1, pba1]];
    var pCgivenB = [[1-C_GIVEN_0, C_GIVEN_0],[1-pcb1, pcb1]];

    var jointB1 = 0;
    for(var a=0;a<2;a++){
      for(var c=0;c<2;c++){
        jointB1 += pA[a]*pBgivenA[a][1]*pCgivenB[1][c];
      }
    }

    var msgAtoB = [];
    for(var b=0;b<2;b++){ msgAtoB[b] = pA[0]*pBgivenA[0][b] + pA[1]*pBgivenA[1][b]; }
    var msgCtoB = [1, 1];
    var mB = [msgAtoB[0]*msgCtoB[0], msgAtoB[1]*msgCtoB[1]];
    var norm = mB[0]+mB[1];
    var pB1_msg = mB[1]/norm;

    var pB1_brute = jointB1;
    document.getElementById('ch34_brute').textContent = pB1_brute.toFixed(4);
    document.getElementById('ch34_msg').textContent = pB1_msg.toFixed(4);
    document.getElementById('ch34_match').textContent = Math.abs(pB1_brute-pB1_msg) < 1e-9 ? 'yes' : 'no';

    var plot = SL.makePlot(document.getElementById('ch34_canvas'), {l:50,r:20,t:20,b:40});
    plot.setDomain([0,3],[0,1]); plot.clear();
    var ctx = plot.ctx;
    ctx.strokeStyle = SL.colors.inkSoft; ctx.lineWidth = 2;
    function node(x, label, val){
      ctx.beginPath(); ctx.arc(plot.px(x), plot.py(0.5), 26, 0, 2*Math.PI);
      ctx.fillStyle = SL.colors.accent2; ctx.fill();
      ctx.fillStyle = SL.colors.ink; ctx.textAlign='center'; ctx.font='13px -apple-system,sans-serif';
      ctx.fillText(label, plot.px(x), plot.py(0.5)+5);
    }
    ctx.beginPath(); ctx.moveTo(plot.px(0.5)+26, plot.py(0.5)); ctx.lineTo(plot.px(1.5)-26, plot.py(0.5)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(plot.px(1.5)+26, plot.py(0.5)); ctx.lineTo(plot.px(2.5)-26, plot.py(0.5)); ctx.stroke();
    node(0.5,'A'); node(1.5,'B'); node(2.5,'C');
    ctx.fillStyle = SL.colors.inkSoft; ctx.font='11px -apple-system,sans-serif';
    ctx.fillText('P(A=1)='+pa1.toFixed(2), plot.px(0.5), plot.py(0.5)+48);
    ctx.fillText('P(B=1|A)='+pba1.toFixed(2), plot.px(1.5), plot.py(0.5)+48);
    ctx.fillText('P(C=1|B)='+pcb1.toFixed(2), plot.px(2.5), plot.py(0.5)+48);
  }
  ['ch34_pa','ch34_pba','ch34_pcb'].forEach(function(id){
    document.getElementById(id).addEventListener('input', update);
  });
  update();
});
</script>

## Hidden Markov models: forward-backward is belief propagation on a chain

A **hidden Markov model** (HMM) has hidden states $S_1,\dots,S_T$ forming a chain - exactly the **Markov chain** of Chapter 24, where $P(S_t \mid S_{t-1})$ is the transition - and observations $O_t$ depending only on $S_t$. The graph is a chain, so sum-product applies, and under its HMM name it is the **forward-backward algorithm**:

$$\alpha_t(s) \propto P(O_t\mid S_t{=}s)\sum_{s'} P(S_t{=}s\mid S_{t-1}{=}s')\,\alpha_{t-1}(s'), \qquad \beta_t(s) = \sum_{s'} P(O_{t+1}\mid s')\,P(s'\mid s)\,\beta_{t+1}(s').$$

**Reading it piece by piece:**
- **$\alpha_t(s)$** - the forward message: the joint probability of the observations up to time $t$ and being in state $s$ at time $t$.
- **$P(O_t\mid S_t{=}s)$** - the emission likelihood of today's observation given the current hidden state.
- **$\sum_{s'}$** - summing over all possible previous states $s'$, marginalizing the past into one number.
- **$P(S_t{=}s\mid S_{t-1}{=}s')$** - the transition probability from previous state $s'$ to current state $s$, the Markov chain's edge.
- **$\alpha_{t-1}(s')$** - the forward message from the previous step, the entire past history already summarized.
- **$\beta_t(s)$** - the backward message: the likelihood of all observations *after* time $t$ given state $s$ now.
- **$P(O_{t+1}\mid s')$ and $P(s'\mid s)$** - the next emission and transition, running the same recursion in reverse so the two passes together give $P(S_t \mid \text{all observations})$. is the **forward algorithm** (computing the likelihood of the observation sequence in $O(T \cdot |S|^2)$ time instead of enumerating all $|S|^T$ state sequences), and the forward-backward product renormalized gives $P(S_t \mid \text{all observations})$. Historically, HMMs, HMMs' forward-backward, and Pearl's belief propagation were developed separately across speech recognition and AI; the graphical-model view reveals they are the same recursion. This is the **dynamic programming** pattern you saw in computing edit distance or Fibonacci numbers: solve overlapping subproblems once, cache the result.

## Modern links: factor graphs, CRFs, graph neural networks

- **Factor graphs** - bipartite graphs with variable and factor nodes - unify directed and undirected models and are the clean substrate for generic sum-product and max-product code.
- **Conditional random fields** (CRFs) are undirected graphical models of $P(Y \mid X)$ for structured labels (tagging a sequence, segmenting an image). They are the discriminative counterpart to HMMs and dominated sequence labeling before deep learning.
- **Graph neural networks** (GNNs) are learned message passing: replace the fixed sum-product update with a learned $\phi$, and messages become $\mathbf{h}_v^{(k+1)} = \text{update}\big(\mathbf{h}_v^{(k)}, \sum_{u\in N(v)} \text{msg}(\mathbf{h}_u^{(k)})\big)$. The fixed belief-propagation recursion and the learned GNN layer are the same architecture; one has hand-specified messages, the other has trained ones.

## Worked example - by hand, larger

Consider a 3-node chain $A \to B \to C$, all binary ($0$ or $1$), with local CPTs:

- $P(A=0)=0.6$, $P(A=1)=0.4$.
- $P(B \mid A)$: $P(B{=}0\mid A{=}0)=0.8$, $P(B{=}1\mid A{=}0)=0.2$; $P(B{=}0\mid A{=}1)=0.3$, $P(B{=}1\mid A{=}1)=0.7$.
- $P(C \mid B)$: $P(C{=}0\mid B{=}0)=0.9$, $P(C{=}1\mid B{=}0)=0.1$; $P(C{=}0\mid B{=}1)=0.4$, $P(C{=}1\mid B{=}1)=0.6$.

We want the marginal $P(B)$. **Method 1 - brute-force joint.** Enumerate all 8 joint configurations $P(A,B,C)=P(A)P(B\mid A)P(C\mid B)$, then sum over $A$ and $C$:

| $A$ | $B$ | $C$ | $P(A,B,C)$ |
|-----|-----|-----|-----------|
| 0 | 0 | 0 | $0.6\cdot0.8\cdot0.9 = 0.432$ |
| 0 | 0 | 1 | $0.6\cdot0.8\cdot0.1 = 0.048$ |
| 0 | 1 | 0 | $0.6\cdot0.2\cdot0.4 = 0.048$ |
| 0 | 1 | 1 | $0.6\cdot0.2\cdot0.6 = 0.072$ |
| 1 | 0 | 0 | $0.4\cdot0.3\cdot0.9 = 0.108$ |
| 1 | 0 | 1 | $0.4\cdot0.3\cdot0.1 = 0.012$ |
| 1 | 1 | 0 | $0.4\cdot0.7\cdot0.4 = 0.112$ |
| 1 | 1 | 1 | $0.4\cdot0.7\cdot0.6 = 0.168$ |

Sum the whole column: $0.432+0.048+0.048+0.072+0.108+0.012+0.112+0.168 = 1.000$ ✓. Now group by $B$. For $B=0$: $0.432+0.048+0.108+0.012 = 0.600$. For $B=1$: $0.048+0.072+0.112+0.168 = 0.400$. So $P(B=0)=0.6$, $P(B=1)=0.4$.

**Method 2 - message passing.** Send the message from $A$ to $B$: $m_{A\to B}(b) = \sum_a P(a) P(b\mid a)$. For $b=0$: $0.6\cdot0.8 + 0.4\cdot0.3 = 0.48+0.12 = 0.600$. For $b=1$: $0.6\cdot0.2 + 0.4\cdot0.7 = 0.12+0.28 = 0.400$. Send the message from $C$ to $B$: $m_{C\to B}(b) = \sum_c P(c\mid b)$. Because $C$'s factor is a normalized distribution over $c$, this sums to 1 for each $b$ - $m_{C\to B}(0)=1$, $m_{C\to B}(1)=1$. (This is the sum-product rule: the message is the sum over the clique factor's other variable.)

Now the belief at $B$ is the product of incoming messages, normalized:

$$P(B=b) \propto m_{A\to B}(b)\cdot m_{C\to B}(b): \quad P(B=0)\propto 0.600\cdot1=0.600,\quad P(B=1)\propto0.400\cdot1=0.400.$$

Normalizing: total $= 1.000$, so $P(B=0)=0.6$ and $P(B=1)=0.4$ - **identical to the brute-force joint sum**, and we never touched $C$'s internal state or enumerated 8 configurations. The message passing did $O(\text{edges})$ work where brute force did $O(2^3)$.

**But watch what $C$ did.** Had $C$'s factor not been a normalized conditional - say an undirected factor $\phi(B,C)$ whose entries don't sum to 1 - then $m_{C\to B}(0)$ and $m_{C\to B}(1)$ would both be scaled by that unnormalized constant, and the final normalizing denominator would no longer be the trivially-computed total. That denominator is $Z$, the partition function, and for a general loopy graph computing it is the hard part. On this chain it was free; on an arbitrary graph it is #P-hard.

**Takeaway: on a tree, local messages reconstruct the exact global marginal with linear work, and the partition function only becomes a genuine computational wall once the graph - or the factors - stop being nicely normalized trees.**

## Check your understanding

- Why does the chain rule alone not give you a tractable model, while the Bayesian-network factorization does - what exactly does the independence assumption remove?
- In the worked example, why did the message from $C$ to $B$ come out as the constant $1$ for both values of $b$, and what would change if $C$'s factor were unnormalized?
- Forward-backward is described as belief propagation on a chain. What are the "messages" in the forward pass, and what does the backward pass add that the forward pass alone does not provide?

## Exercises

### Exercise 1 - Marginal by message passing on a 2-node chain

A chain $A \to B$ with $P(A=0)=0.7$, $P(A=1)=0.3$, and $P(B=0\mid A=0)=0.5$, $P(B=1\mid A=0)=0.5$, $P(B=0\mid A=1)=0.1$, $P(B=1\mid A=1)=0.9$. Compute $P(B=0)$ and $P(B=1)$ by message passing, then verify by brute-force enumeration.

<details><summary>Show solution</summary>

Message $m_{A\to B}(b) = \sum_a P(a)P(b\mid a)$. For $b=0$: $0.7\cdot0.5 + 0.3\cdot0.1 = 0.35+0.03=0.38$. For $b=1$: $0.7\cdot0.5+0.3\cdot0.9=0.35+0.27=0.62$. Since $B$ has no other neighbors, $P(B=b)=m_{A\to B}(b)$ directly (already sums to 1). So $P(B=0)=0.38$, $P(B=1)=0.62$.

Brute force: joint table $(A,B)$: $(0,0)=0.7\cdot0.5=0.35$, $(0,1)=0.35$, $(1,0)=0.3\cdot0.1=0.03$, $(1,1)=0.27$. Column sum $=1.00$ ✓. Grouping by $B$: $B=0$ gives $0.35+0.03=0.38$; $B=1$ gives $0.35+0.27=0.62$. Matches.

</details>

### Exercise 2 - Sum-product versus brute force, counting operations

For a chain of $n$ binary variables, how many multiplications does brute-force marginal computation of an interior variable require versus a single forward-backward pass? Compute both for $n=20$ and comment.

<details><summary>Show solution</summary>

Brute force enumerates all $2^{n-1}$ assignments of the *other* variables for each of the $2$ values of the target, evaluating a product of $n$ factors each time - roughly $2 \cdot 2^{n-1}\cdot n = n\,2^n$ operations. For $n=20$: $20\cdot 2^{20} \approx 20 \times 1{,}048{,}576 \approx 2.1\times10^7$.

Message passing does one left-to-right and one right-to-left pass, each adding $O(2^2)$ per edge: total on the order of $n$ edges $\times$ 4 sum-multiply terms $\approx 4n = 80$ operations. For $n=20$ that is $80$ versus $2.1\times10^7$ - a factor of roughly $260{,}000$, and the gap grows exponentially in $n$. This is exactly why dynamic programming / message passing is the whole game: the same answer, polynomial instead of exponential work.

</details>

### Exercise 3 - Conceptual: why loopy graphs break exactness

Belief propagation is exact on trees. Explain in conceptual terms why running the same recursion on a graph with a cycle (a "loopy" graph) gives, in general, only an approximate answer.

<details><summary>Show solution</summary>

On a tree there is exactly one simple path between any two nodes, so the message from $u$ to $v$ summarizes information that never re-enters through another route - the messages "partition" the graph cleanly, and the product of incoming messages is the exact marginal. With a cycle, information can travel around the loop and return to a node, so a message partially contains evidence that already arrived by the other arm of the cycle. The product then effectively **double-counts** that shared evidence, so the node's belief is no longer the true marginal. The recursion still runs and often gives a good approximation (this is **loopy belief propagation**, widely used in error-correcting codes and vision), but it has no exactness guarantee, and it may not even converge; only on trees is the partition of information perfect and the answer provably exact.

</details>

## Why This Matters - In AI and In Everyday Life

**In AI / machine learning:**

- **Autoregressive language models are chain-structured graphical models**: $P(\text{tokens}) = \prod_t P(x_t \mid x_{<t})$ is exactly the chain-rule factorization of a Bayesian network, and the transformer's forward pass computes these conditionals left to right.
- **Graph neural networks are learned message passing**: the fixed sum-product update is replaced by learned message and update functions, which is why GNNs inherit both the strengths (locality, permutation invariance) and the weaknesses (over-squashing over long paths) of belief propagation.
- **Sequence models and structured prediction**: HMMs, CRFs, and their deep successors all rely on forward-backward / Viterbi dynamic programming to handle sequences in polynomial rather than exponential time.
- **The partition function is the central obstacle in modern generative models**: normalizing flows, energy-based models, and undirected deep models all fight the same intractable $Z$ that this chapter met on a tiny chain, which is why we approximate it with MCMC (Chapter 24) or design it away by construction.

**In everyday, non-AI life:**

- **Diagnosis and fault-finding**: reasoning from symptoms (observed) back to a failing component (hidden) through a network of components that influence one another is Bayesian-network inference, whether or not the mechanic draws the DAG.
- **Detective reasoning over a timeline**: each clue conditions on the previous state, and updating belief as clues arrive in sequence is forward-backward over a chain of hypotheses.

## What's next

Chapter 35 pivots from *representing* a distribution to *optimizing* a model of it. We revisit the bias-variance picture of Chapter 16, then watch it famously break: the **double descent** curve, where test error drops, spikes at the interpolation threshold, and then *falls again* as models grow far past the number of data points - the phenomenon this chapter's exact-inference trees and Chapter 22's generalization theory both help frame.

<!-- chapter 34 -->