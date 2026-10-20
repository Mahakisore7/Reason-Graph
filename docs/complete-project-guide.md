# Reason-Graph: Complete Project Guide

This document explains the full project from first principles: the motivation, the math, the architecture, the code structure, the data, the experiments, and how the repo turns a raw statement into a probabilistic hallucination score.

The project is a research-engineering repo for probabilistic hallucination detection. It reproduces and extends BTProp (Belief Tree Propagation) using open-source models and a clear Python implementation.

---

## 1. What problem the project solves

Large language models often answer fluently while being wrong. The problem is not just “does the model sound confident?” but “how likely is the statement actually true?”

The repo’s central idea is:

- a statement should not be judged in isolation,
- the claim should be placed inside a network of related claims,
- related statements constrain one another,
- the truth of the whole structure can be combined in one probabilistic score.

This is exactly the idea behind BTProp:

1. Expand a claim into related claims (decomposition, premises, corrections).
2. Ask an LLM for its confidence on each claim.
3. Label each relationship with NLI (entailment, contradiction, equivalence, etc.).
4. Combine all this evidence in a tree-based probabilistic model.
5. Produce a corrected posterior probability that the original claim is true.

The repo adds two extensions:

- answer-level graphs with loopy belief propagation,
- a surrogate confidence model that predicts LLM confidence from text and reduces repeated LLM calls.

---

## 2. Why a naïve single confidence is not enough

A standard LLM call might be:

- “Is the following statement true or false?”

This gives a single score like 0.93. But the model can be confidently wrong. The repo treats raw confidence as only a noisy observation, not the final answer.

The deeper idea is:

P(true | surrounding evidence) instead of P(true | raw one-shot answer)

This is the project’s core reason for using a structured belief tree.

---

## 3. High-level pipeline

The overall flow is:

1. Take a sentence or answer.
2. Build a belief tree around it.
3. Generate child claims using the LLM.
4. Use NLI to label each child-parent relationship.
5. Convert the tree into a probabilistic model.
6. Compute the posterior probability that the root claim is true.
7. For answer-level scoring, connect sentence trees into one graph and run loopy BP.
8. Optionally replace some LLM confidence calls with a surrogate model.

Conceptually:

P(root true | observable confidences, generated claims, NLI relationships)

---

## 4. Core mathematical idea

### 4.1 Hidden binary state

Each claim u has a hidden variable:

Z_u ∈ {True, False}

The LLM provides an observed confidence score S_u. The model treats this as noisy evidence about the hidden truth of the claim.

The repo’s probability model is:

- emission model: P(S_u | Z_u)
- transition model: P(children_truth | parent_truth)
- posterior: P(Z_root = True | all observed scores)

### 4.2 Tree posterior

For a tree, the exact inference computes:

beta(z, u) = P(all observed scores in the subtree of u | Z_u = z)

For leaves:

beta(z, u) = P(S_u | Z_u = z)

For internal nodes:

beta(z, u) = P(S_u | Z_u = z) × sum over child assignments [ P(children | Z_u = z) × ∏_children beta(child_truth) ]

Then Bayes’ rule gives the root posterior:

P(Z_root = True | observations) =
  beta(True, root) × P(Z_root = True)
  / [ beta(True, root) × P(Z_root = True) + beta(False, root) × P(Z_root = False) ]

This is the exact root score used by the project.

### 4.3 Why tree relationships matter

If a child claim is:

- equivalent to the parent: they usually have the same truth value,
- entailing the parent: the child implies the parent,
- reverse entailment: the parent implies the child,
- contradictory: one is likely false if the other is true,
- neutral: unrelated.

These relationships define the transition model, which changes the posterior calculation.

---

## 5. The project architecture at a glance

The repository is split into modules that correspond to the research pipeline:

- `src/llm/` — LLM clients and prompts
- `src/tree/` — belief-tree generation and NLI labeling
- `src/inference/` — exact tree inference and loopy graph propagation
- `src/surrogate/` — cheap surrogate confidence scoring
- `src/eval/` — metrics
- `src/pipeline.py` — end-to-end verification logic
- `scripts/` — dataset processing, training/calibration, evaluation, figures
- `tests/` — unit tests for each core component

---

## 6. The LLM abstraction layer

The repo’s design is deliberately modular. The probabilistic logic does not depend on whether the model is OpenAI, Hugging Face, or Groq.

This is defined in `src/llm/base.py`.

The protocol is basically:

- `complete(prompt, ...)` → free-form text
- `complete_samples(prompt, n, ...)` → multiple generated samples
- `true_false_confidence(statement)` → score in [0,1] for the claim’s truth

This design allows the rest of the code to stay independent from model-specific details.

### 6.1 `HuggingFaceClient`

`src/llm/hf_client.py` runs a local open-source model and reads next-token probabilities directly.

This is the most faithful implementation of the BTProp-style method because it computes the model’s internal probability mass for “True” vs “False”.

The helper function is conceptually:

P(True) = sum(probabilities for true variants) / (sum(true variants) + sum(false variants))

It uses token IDs for variants like:

- “True”, “ true”, “TRUE”, etc.
- “False”, “ false”, “FALSE”, etc.

The project treats this as the strongest/cleanest confidence estimate.

### 6.2 `OpenAIClient`

`src/llm/openai_client.py` uses the OpenAI API and reads log probabilities from the completion response. This can also approximate the paper’s method, as long as the backend exposes token-level probabilities.

### 6.3 `GroqClient`

`src/llm/groq_client.py` is a free, practical fallback. But Groq has a limitation: it does not expose logprobs on the models used here. So it asks the model to emit its confidence as a bare number in [0, 1].

This is weaker and explicitly marked as provisional in the repo.

### 6.4 `CachingLLMClient`

`src/llm/cache.py` wraps any backend and stores every LLM result on disk. This is essential because:

- repeated prompts are avoided,
- runs are reproducible,
- an interrupted run can resume,
- the same evaluation can be replayed offline without rerunning a model.

The cache key includes the model tag and the exact prompt arguments, so changing the prompt or backend changes the cache identity.

---

## 7. Prompt engineering: how the belief tree is generated

The prompt templates live in `src/llm/prompts.py`.

The prompts are not arbitrary; they mirror the official BTProp-style tree-generation process. The README explicitly calls them “copied from the official BTProp repository.”

The repo uses prompts for:

- decomposition,
- strategy selection,
- premise generation,
- correction generation,
- confidence scoring.

### 7.1 Decomposition prompt

This prompt asks the LLM to split a statement into distinct factual claims. The extracted claims become child nodes of the root.

The code function is:

- `decomposition_strategy(llm, statement)`

It reads lines starting with `Claim` and extracts the text after the label.

The repo filters out nuisance outputs such as:

- `[ ... ]` placeholders,
- “No answer”, “I don’t know”, etc.,
- obviously non-checkable claims.

### 7.2 Strategy selection

The standard BTProp style is not always “premises” or “correction” only. The model is asked which strategy is most suitable. The repo code does this in `select_strategies(...)`.

If the model says “perturbation”, it tends to choose corrections. If it says “logical relation”, it goes for premises.

The repo also allows both in a combined route.

### 7.3 Premises generation

The LLM is asked for supportive premises. The code collects these and keeps only valid ones.

The approach is:

- create multiple premise samples,
- parse lines with `Premise`,
- filter out things like “No supportive premises applicable.”

### 7.4 Correction generation

This is the “perturbation” strategy. It asks for a key question and then generates alternative versions of the original statement that fit that question.

The repo does:

- ask for a “wh-question” around the statement,
- sample several answers,
- revise the statement to match each answer,
- keep only coherent revisions,
- treat them as candidate corrections.

Corrections are treated specially because they are alternative versions of the same claim, not independent facts.

---

## 8. Relationship labels: NLI and logic

The project uses NLI to label the relationship between a generated child claim and its parent claim.

This is implemented in `src/tree/nli_labeler.py` and `src/tree/relationships.py`.

### 8.1 Why both directions matter

A single NLI direction only says something like:

- premise entails hypothesis,
- premise contradicts hypothesis,
- premise is neutral with respect to hypothesis.

But the direction matters.

The repo runs NLI in both directions:

- `forward = directional.label_direction(child, parent)`
- `backward = directional.label_direction(parent, child)`

Then it combines them as:

- forward entailment + backward entailment → equivalence
- forward entailment only → entailment
- backward entailment only → reverse entailment
- either contradiction → contradiction
- otherwise → neutral

This is how the repo decides whether a child supports, contradicts, or is unrelated to the parent.

### 8.2 Relationship enum

Defined in `src/tree/relationships.py`:

- `EQUIVALENCE`
- `ENTAILMENT`
- `REVERSE_ENTAILMENT`
- `CONTRADICTION`
- `NEUTRAL`

The relationship determines how likely a child is to be true given the parent’s truth state.

### 8.3 Transition probabilities

The repo sets default transition parameters in a dictionary. These encode the prior probability relationship between parent and child:

- equivalence: (1.0, 0.0)
- entailment: (0.5, 0.0)
- reverse entailment: (1.0, 0.5)
- contradiction: (0.0, 0.5)

Interpretation:

- if the parent is true, then a child in entailment relation has some probability of being true,
- if the parent is false, a child cannot be true in the entailment direction,
- contradiction means a true parent makes the child very likely false,
- equivalence means the two share the same truth state almost always.

These are model assumptions used by the HMT transition model.

---

## 9. The raw belief tree data structure

The belief tree is created in `src/tree/builder.py`.

Each node is a `RawNode` with fields such as:

- `node_id`
- `statement`
- `confidence`
- `parent_id`
- `relationship_to_parent`
- `strategy`
- `children`

The tree as a whole is `RawBeliefTree` with:

- `nodes: dict[str, RawNode]`
- `root_id`

The builder works breadth-first, with maximum depth control. It marks each node’s strategy as one of:

- `root`
- `decomposition`
- `premise`
- `correction`

A key implementation detail is that:

- root nodes can generate decomposition children,
- each child is conditionally added only if the NLI relationship is not neutral,
- the node confidence is set directly from the model,
- correction children get fixed confidence 1.0.

---

## 10. The exact tree inference engine

This is in `src/inference/hmt.py`.

### 10.1 Node and emission abstraction

Each `TreeNode` contains:

- `node_id`
- `score`
- `children`

The emission model is a protocol:

- `prob(score, truth)`

This determines how the observed score should be interpreted under the hidden truth value.

The repo implements:

- `LookupEmission` for a table-based calibration,
- `LogisticEmission` for a smooth logistic model.

### 10.2 Transition models

The repo implements:

- `IndependentChildTransition`
- `DecompositionTransition`

These apply to different child-generation strategies.

#### Independent children

For premises/corrections, the children are treated as conditionally independent given the parent. This is expressed by:

P(children_truth | parent_truth) = ∏_child P(child_truth | parent_truth)

#### Decomposition

When a parent is decomposed into multiple claims, the child claims jointly express the parent. If the parent is true, all of its decomposition children must be true. If the parent is false, at least one child is false.

That is why the decomposition transition is different from the independent-child transition.

### 10.3 Compute-beta recursion

This is the core exact inference algorithm. It recursively computes the contribution of each subtree to the likelihood of the root.

The function `compute_beta(...)` does this exactly.

The code uses the product over all possible combinations of child truth values.

This is computationally expensive for large trees but exact on trees.

### 10.4 Posterior

The root posterior is returned by `posterior_root_true(tree)`.

This function combines the two candidate likelihoods with a prior and normalizes.

---

## 11. Calibration: turning raw confidence into probabilistic evidence

The repo explicitly separates two ideas:

- raw LLM confidence,
- calibrated evidence for inference.

This matters because LLM confidence is not necessarily calibrated. A score of 0.9 does not automatically mean “90% likely true.”

The repo fits an emission model that maps a raw confidence into probabilities of seeing that score under true and false states.

### 11.1 Why calibration is needed

An LLM may give high confidence to some false statements. The tree model needs to understand how confidence behaves in practice.

The project fits calibration from labelled data. The code is in `scripts/fit_emission_model.py`.

### 11.2 The binned table model

A simple calibration is a table of confidence bins. The repo uses the same bins as the paper-like setup:

- 0.19
- 0.39
- 0.69
- 0.89

That means scores are bucketed into the ranges:

- [0, 0.19]
- (0.19, 0.39]
- (0.39, 0.69]
- (0.69, 0.89]
- (0.89, 1]

The bucket table stores the empirical probability that a score in that range came from a true vs false statement.

### 11.3 Logistic calibration

The repo also fits a logistic emission model with `LogisticRegression`:

logit(P(true | score)) = slope × logit(score) + intercept

This is implemented by `fit_logistic(...)` in `scripts/fit_emission_model.py` and by `LogisticEmission` in `src/inference/hmt.py`.

The model uses the fitted curve to obtain:

P(Z = True | score)

and then converts it into the emission model needed by HMT inference.

---

## 12. Data and dataset split

The repo uses the FELM-Science dataset from HuggingFace.

The README and project log describe the dataset as the benchmark used by the paper. It contains AI-written answers with human-labeled sentence truth values.

The code loads raw `all.jsonl` from `hkust-nlp/felm` and filters only the science domain.

### 12.1 Cleaning and filtering

The code removes:

- short non-claim snippets,
- duplicate answers,
- list-intro segments that are not actual claims,
- obviously unrelated filler segments.

This is implemented in `scripts/fit_emission_model.py` with functions like:

- `is_claim(...)`
- `split_answers(...)`

### 12.2 Fit/test split

Crucially, the split is performed by whole answer, not by sentence. This avoids leakage because sentences from the same answer are topical and share language patterns.

The repo splits the answers into:

- fit half: used to build calibration and choose methodological choices,
- test half: used only for evaluation.

This matches the project’s “no leaking fit/test information” rule.

---

## 13. The end-to-end verification pipeline

The central API is in `src/pipeline.py`.

### 13.1 `verify_statement(...)`

This builds a belief tree and computes the posterior probability that the root statement is true.

It returns a `VerificationResult` containing:

- statement,
- posterior_true,
- is_trustworthy,
- tree_size,
- raw_tree.

### 13.2 `verify_answer(...)`

This goes beyond tree-level scoring. It builds one graph for the whole answer, where:

- each sentence gets its own tree,
- sentence roots are connected by cross-sentence NLI edges,
- the whole graph is processed with loopy BP,
- each sentence gets a final marginal probability.

It returns:

- sentences,
- posterior_true per sentence,
- tree_only scores per sentence,
- links,
- convergence flag,
- the trees.

This design gives a direct comparison between tree-only and graph-level scoring.

---

## 14. The answer-graph extension: one graph per answer

This is the project’s first major novelty.

The repo’s README emphasizes that the paper evaluates each sentence independently, but the project treats related sentences in the same answer as a connected probabilistic system.

The approach:

1. Build a tree for each sentence.
2. Label NLI relationships between sentence roots.
3. Add these as cross links between sentence-level root variables.
4. Build a factor graph.
5. Use loopy belief propagation to estimate posterior marginals for all sentence roots.

### 14.1 Cross links

Cross links depend on the chosen rule. The repo supports rules like:

- `any`
- `strict`
- `no_contradiction`

The default is `no_contradiction`, which was chosen on the fit data as the best performing rule.

This rule drops contradiction links and keeps support links, because raw contradictions between different sentences in the same answer often over-trigger and hurt the graph.

### 14.2 Factor graph representation

In `src/inference/loopy.py`, a factor graph is built from variables and factors.

A factor is a table over the joint truth values of its connected variables. The project uses three kinds of factors:

- unary evidence factors: P(score | hidden truth)
- tree joint factors: same parent-child logic as the HMT tree transition
- pairwise cross links: a relationship-specific factor between sentence roots

This is the graph version of the same probabilistic model.

### 14.3 Loopy BP update rule

Loopy belief propagation is a message-passing algorithm on a factor graph.

For a factor f connected to variable v, the message is:

m_{f→v}(z_v) ∝ sum_{z_f \ z_v} f(z_f) ∏_{w ∈ neighbors(f)\{v}} m_{w→f}(z_w)

For variable v to factor f:

m_{v→f}(z_v) ∝ ∏_{g ∈ neighbors(v)\{f}} m_{g→v}(z_v)

The repo uses synchronous updates with damping and convergence tolerance. This is the actual algorithm in `run_loopy_bp(...)`.

It reports:

- whether the graph converged,
- how many iterations were used,
- the final variable marginals.

### 14.4 Important design choice

The project does not assume convergence. On graphs with cycles, there is no guarantee. So convergence is measured and reported, not assumed.

---

## 15. The surrogate confidence scorer

This is the second major extension.

The repo’s README frames it as the cost-saving novelty: a large fraction of the question-answering budget is spent getting confidence scores for every node in every tree. The project replaces some of those scores with a cheap learned estimator.

### 15.1 The idea

Given a statement text, predict the LLM’s confidence for that statement. The surrogate predicts the LLM’s own score, not the human label.

Thus, the surrogate learns:

text → confidence from the model

not:

text → truth label

### 15.2 Hashing embedder

The embedder is in `src/surrogate/embedder.py`.

It uses a `HashingVectorizer` with:

- word 1-grams and 2-grams,
- 4096 features,
- `alternate_sign=False`,
- `norm='l2'`.

This is cheap, does not require a pre-trained embedding model, and is fast to fit.

### 15.3 Ridge regressor

`SurrogateConfidenceModel` in `src/surrogate/model.py` fits a `Ridge` regressor:

confidence ≈ w^T x + b

where x is the hashed text feature vector.

It clips predictions into [0,1], because a confidence score must be a valid probability-like number.

### 15.4 No leakage

The project avoids leaking test-answer information by using group-based cross-fitting. The data is split into folds by answer, and each answer’s nodes are predicted by a model trained on other answers only.

This prevents the surrogate from learning from the same answer it is asked to score.

### 15.5 Delegation curve

The repo measures the effect of replacing a fraction of LLM confidence calls with the surrogate. It test several fractions:

- 0%
- 25%
- 50%
- 75%
- 100%

and compares the resulting AUROC. This is how the repo quantifies the cost/quality tradeoff.

---

## 16. Evaluation and metrics

The evaluation logic is in `src/eval/metrics.py`.

The project uses:

- AUROC
- AUC-PR for the false class
- F1 for false detection

### 16.1 AUROC

AUROC measures how well the model ranks true statements above false ones.

It is invariant to threshold choice and is a classic ranking metric.

### 16.2 AUC-PR

AUC-PR is particularly important because the false class is rare. The repo computes average precision for finding false statements using the transformed score:

hallucination score = 1 - P(true)

### 16.3 F1 and cross-fitted threshold

The repo reports F1 with a threshold on P(true). To avoid threshold overfitting, it uses cross-fitted thresholds across answer groups.

The logic is:

- split the data into folds by answer group,
- pick the threshold that maximizes F1 on the training folds,
- apply it to the held-out fold,
- aggregate predictions across all folds.

This prevents the threshold from being selected on the same statements it is supposed to evaluate.

---

## 17. The scripts and how they fit the full workflow

The `scripts/` directory contains the project’s experiment pipeline.

### 17.1 `fit_emission_model.py`

Fits the calibration model from labelled data. This is the “learn how confidence maps to truth probability” step.

It:

- loads FELM science answers,
- cleans them,
- splits into fit and test halves,
- gets LLM confidence on each segment,
- fits the binned or logistic emission model.

### 17.2 `evaluate_felm.py`

Evaluates the tree-based BTProp reproduction on the test half. It builds a tree for each statement, computes the posterior, and measures AUROC.

It compares:

- raw confidence,
- calibration-only,
- tree+calibration.

### 17.3 `evaluate_graph.py`

Builds one graph per answer and runs loopy BP. It reports:

- per-sentence graph scores,
- tree-only scores,
- convergence,
- AUROC differences.

### 17.4 `evaluate_surrogate.py`

Measures how a surrogate can replace LLM confidence calls on some or all nodes.

It tracks:

- surrogate error vs LLM confidence,
- effect on AUROC when fraction of nodes are delegated,
- difference between any-node delegation and children-only delegation.

### 17.5 `verify.py`

This is the demo script.

It lets you check:

- one statement with the full tree method,
- multiple sentences in one answer with the graph method.

### 17.6 `report_metrics.py` and `make_figures.py`

These scripts summarize the results and create the figures used in the project documentation and README.

---

## 18. The exact code path for a single claim

Here is the exact flow in code terms:

1. `build_belief_tree(llm, labeler, statement, max_depth=1)`
   - in `src/tree/builder.py`
   - generates the root node and its descendants,
   - calls decomposition / premises / correction strategies,
   - applies NLI labeling,
   - returns a `RawBeliefTree`.

2. `build_hidden_markov_tree(raw, calibration)`
   - in `src/pipeline.py`
   - converts the raw tree into a `HiddenMarkovTree`,
   - assigns each node a score based on the calibration’s `to_score` function,
   - attaches the proper emission and transition models.

3. `posterior_root_true(tree)`
   - in `src/inference/hmt.py`
   - computes the exact posterior probability of the root.

This is the core mathematical pipeline of the project.

---

## 19. The exact code path for an answer graph

For an answer with multiple sentences:

1. build one tree per sentence via `build_belief_tree(...)`
2. label cross-sentence links by `label_cross_links(...)`
3. build a `FactorGraph` using `build_answer_graph(...)`
4. run `run_loopy_bp(...)`
5. use the resulting marginals as each sentence’s posterior probability

The key functions are in `src/pipeline.py`.

The graph is a generalization of the belief-tree method and is the project’s extension beyond the paper.

---

## 20. Why the project chooses a graph instead of a single tree

In a single statement, a belief tree is natural. But in a multi-sentence answer, sentence-level facts are connected by support and contradiction. A tree structure is too restrictive because a node may have multiple meaningful parents or neighbors.

So the project creates a factor graph to allow the information to propagate across the whole answer.

This is a real modeling improvement beyond the original paper’s per-sentence independent treatment.

---

## 21. Why the repo is strong engineering-wise

The project is not just a paper prototype. It is structured like a real research codebase:

- common interface for different LLM backends,
- cache layer for reproducibility,
- modular separation between tree-building, inference, evaluation, and experiments,
- deterministic tests for each major component,
- clear calibration/evaluation scripts,
- offline replay support for not rerunning expensive LLM calls,
- experiment outputs saved as JSON for later comparison.

This makes it possible to reproduce and compare approaches cleanly.

---

## 22. What the repo proves in practice

The README and project log summarize the main outcomes:

- raw confidence is a weak baseline,
- tree propagation helps,
- answer graphs can improve performance when the link rules are chosen correctly,
- surrogate scoring can save a large share of confidence calls without destroying performance,
- data cleaning and standalone rewriting are essential,
- using open-source LLMs and local replay is feasible.

So the repo is not just a method demo; it is a functioning research system around a probabilistic idea.

---

## 23. File-by-file mental map

Here is the conceptual map of the most important files:

- `README.md` — project overview and high-level claims.
- `pyproject.toml` — Python package and dependencies.
- `src/llm/base.py` — abstract LLM interface.
- `src/llm/prompts.py` — BTProp prompts.
- `src/llm/hf_client.py` — open-source inference with token probabilities.
- `src/llm/groq_client.py` — free API fallback.
- `src/llm/openai_client.py` — OpenAI API client.
- `src/llm/cache.py` — disk caching.
- `src/tree/strategies.py` — decomposition/premise/correction generation.
- `src/tree/relationships.py` — relationship enum and transition numbers.
- `src/tree/nli_labeler.py` — NLI combination logic.
- `src/tree/builder.py` — raw tree construction.
- `src/inference/hmt.py` — exact tree inference.
- `src/inference/loopy.py` — graph factorization and loopy BP.
- `src/surrogate/embedder.py` — text hashing.
- `src/surrogate/model.py` — surrogate regression model.
- `src/eval/metrics.py` — AUROC / PR / F1.
- `src/pipeline.py` — full verification pipeline.
- `scripts/fit_emission_model.py` — calibration fitting.
- `scripts/evaluate_felm.py` — tree evaluation.
- `scripts/evaluate_graph.py` — answer graph evaluation.
- `scripts/evaluate_surrogate.py` — surrogate evaluation.
- `scripts/verify.py` — demo utility.
- `tests/*.py` — verification tests for each module.

---

## 24. The core takeaway

The project is a probabilistic reasoning system that turns a statement into a small evidence graph, scores the evidence with a model and NLI labels, and returns a calibrated posterior that the statement is true.

It combines:

- language generation,
- logical relationship detection,
- probabilistic inference on trees and graphs,
- calibration from real data,
- and cost reduction through surrogate confidence models.

That is the full conceptual story of Reason-Graph.

---

## 25. How to read the repo in a practical order

If you want to understand the repository in the right order, read in this sequence:

1. `README.md`
2. `docs/project-log/01-big-picture.md`
3. `docs/project-log/02-what-we-built.md`
4. `docs/project-log/03-decisions.md`
5. `src/inference/hmt.py`
6. `src/tree/builder.py`
7. `src/tree/nli_labeler.py`
8. `src/inference/loopy.py`
9. `src/pipeline.py`
10. `scripts/fit_emission_model.py`
11. `scripts/evaluate_felm.py`
12. `scripts/evaluate_graph.py`
13. `scripts/evaluate_surrogate.py`
14. `tests/`

This order follows the project from theory → implementation → evaluation → extension.

---

## 26. Final summary

The project is best understood as an end-to-end probabilistic hallucination detection system built around BTProp, extended to answer-level graphs and cheaper confidence estimates.

It uses:

- LLM-generated structured evidence,
- NLI relationship labels,
- exact tree-level Bayesian inference,
- graph-level loopy BP,
- calibration of raw confidence,
- surrogate confidence approximation,
- dataset-based empirical validation.

This is the project’s complete conceptual and implementation story.
