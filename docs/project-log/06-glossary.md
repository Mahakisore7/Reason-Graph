# Glossary

**Hallucination** — an LLM stating something false as if it were true.

**Claim / statement / segment** — one checkable piece of text. In FELM, a "segment" is one
human-labelled piece of an answer, usually a sentence.

**Belief tree** — a claim at the top, related claims below it as children, possibly with their
own children.

**Decomposition / Premises / Correction** — the three ways we grow children from a claim. See
[01-big-picture.md](01-big-picture.md).

**Confidence score** — the LLM's own estimate (0 to 1) that a claim is true.

**Logprobs** — the probability the model internally gives to each possible next word. The
paper reads the probability of "True" vs. "False" directly. OpenAI exposes this; Groq doesn't.

**Typed-out confidence** — asking the model to write a number from 0 to 1. A weaker substitute
for logprobs (our Groq approach).

**NLI (Natural Language Inference)** — deciding whether sentence A *entails* sentence B (A true
means B true), *contradicts* it (they can't both be true), or is *neutral* (unrelated). We use a
separate small model trained only for this.

**Equivalence / entailment / reverse entailment / contradiction / neutral** — our 5 link types
between a child and its parent. See D9 in [03-decisions.md](03-decisions.md).

**Hidden Markov Tree (HMT)** — the probability model behind the paper. Each claim has a hidden
true/false value we can't see directly. We only see the LLM's confidence score, which is a noisy
signal of it.

**Emission probability / calibration table** — how likely a given confidence score is when the
claim is true vs. when it's false. Learned by counting on labelled data.

**Transition probability** — how likely a child is true given its parent is true (or false).
Depends on the link type.

**Bayes' rule** — the formula for updating a belief when new evidence arrives:
belief after ∝ belief before × how well the evidence fits.

**Belief propagation (upward pass)** — computing, from the leaves up, how well everything
below each claim fits "this claim is true" vs. "this claim is false". The top claim's two values
plus Bayes' rule give the final probability.

**Posterior** — the final, corrected probability that the claim is true, after all evidence.

**DAG (directed acyclic graph)** — like a tree, but a claim can have several parents. Used in
our novelty (Phase 3).

**Loopy belief propagation** — belief propagation repeated until it settles; an approximation
used when the graph isn't a tree.

**AUROC** — accuracy measure. Chance that a randomly chosen true claim scores higher than a
randomly chosen false one. 0.5 = coin flip, 1.0 = perfect.

**Fit set / test set** — data used to learn something (here, the calibration table) vs. data
kept unseen and only used to measure accuracy. Mixing them makes results look better than they
really are.

**Stratified split** — splitting so that both halves keep a similar mix of categories (here:
answers with vs. without errors).
