# Base Paper, Datasets, and Reason-Graph Reproduction

This guide explains the project from the beginning: the problem in the base paper, how BTProp constructs and scores a belief tree, the datasets and preprocessing used in the paper, and what this repository did to reproduce and extend that method.

It is important to distinguish **the paper's experiment** from **this repository's experiment**. The repository follows the paper's central method, but it does not use the same calibration data, preprocessing artifact, tree depth, or evaluation protocol in every respect. Its results are therefore informative comparisons, not a like-for-like rerun of the paper's experiment.

The base paper is Bairu Hou, Yang Zhang, Jacob Andreas, and Shiyu Chang, “A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation,” NAACL 2025. A local PDF copy may be kept outside version control. The repository's implementation decisions are recorded in [03-decisions.md](./project-log/03-decisions.md); the current method, data, and result descriptions are in the [README](../README.md).

---

## 1. The problem addressed by the paper

An LLM can produce a false factual statement while expressing high confidence. A single confidence score is therefore an imperfect measurement of truth:

- it can be **miscalibrated**: statements assigned confidence \(0.9\) are not necessarily true 90% of the time;
- it can be **inconsistent**: related, equivalent, or contradictory claims can receive confidence scores that do not agree with their logical relationship.

The paper proposes **Belief Tree Propagation (BTProp)**. Instead of treating a target claim as an isolated item, BTProp expands it into related claims, scores those claims with the LLM, identifies their logical relationships, and combines their evidence probabilistically.

The quantity of interest is:

\[
P(Z_0=T\mid \text{confidence scores in the belief tree}),
\]

where \(Z_0\) is the hidden truth state of the target/root claim.

The LLM's score is evidence for inference—not itself the final posterior.

---

## 2. Paper method: from a target statement to a belief tree

Let \(u=0\) denote the target statement. A belief tree has:

- one statement at every node \(u\);
- one hidden truth variable \(Z_u\in\{T,F\}\);
- one observed LLM confidence score \(S_u\);
- a parent-child edge whose relationship describes how the claims are logically connected.

The paper recursively repeats two operations until its maximum depth is reached:

1. Generate child statements related to a parent.
2. Determine the logical relationship between each child and its parent.

The experiments in the paper use a maximum tree depth of 2.

### 2.1 Strategy 1: statement decomposition

If a statement contains multiple factual claims, the LLM separates them into individual claims.

For example:

> The Eiffel Tower was completed in 1889 and is in Berlin.

may become:

1. The Eiffel Tower was completed in 1889.
2. The Eiffel Tower is in Berlin.

The parent is modeled as equivalent to the conjunction of its decomposition children:

\[
u \iff \bigwedge_{v\in C(u)}v,
\]

where \(C(u)\) is the set of children of \(u\).

Thus, if the parent is true, every decomposition child must be true:

\[
P\left(Z_{C(u)}=(T,\ldots,T)\mid Z_u=T\right)=1.
\]

If the parent is false, not all decomposition children can be true. The paper spreads the probability over the non-all-true child assignments (see §4.2).

### 2.2 Strategy 2: supportive and contradictory premises

The LLM generates premises that support or challenge the parent claim. NLI determines the relation of each premise to its parent. These children are treated as conditionally independent given the parent truth:

\[
P\left(Z_{C(u)}\mid Z_u\right)
=\prod_{v\in C(u)}P(Z_v\mid Z_u).
\]

This is a modeling assumption that makes inference tractable; it does not assert that natural-language premises are inherently independent.

### 2.3 Strategy 3: statement correction

The LLM identifies a question relevant to the statement, samples answers, and produces corrected versions of the original statement. A correction that contradicts the original can provide evidence against the original.

The paper assigns these correction nodes confidence score \(S=1\) instead of making a separate LLM confidence query. It uses special emission probabilities:

\[
P(S=1\mid Z=T)=0.8,\qquad P(S=1\mid Z=F)=0.2.
\]

The paper does not expand correction nodes further.

### 2.4 Strategy choice

Decomposition is used for multi-claim statements. For non-decomposed statements, the LLM chooses between supportive/contradictory premises, corrections, or both. The paper provides prompts and examples for these steps.

---

## 3. Paper method: determining logical relationships

The paper uses a natural-language-inference model to label a child-parent pair. A directional NLI model classifies a premise/hypothesis pair as:

- **entailment**;
- **contradiction**;
- **neutral**.

The pair is checked in both directions. This distinguishes the two entailment directions:

- **equivalence:** both directions entail;
- **entailment:** one direction entails and the reverse is neutral;
- **reverse entailment:** the other direction entails and the reverse is neutral;
- **contradiction:** the pair is contradictory;
- **neutral:** neither direction entails or contradicts.

Completely neutral children are removed from the belief tree.

**Direction convention in this repository:** its tree builder calls the labeler as `label(child, parent)`. Therefore the `ENTAILMENT` value in the repository means the child entails the parent. This convention is explicit in [relationships.py](../src/tree/relationships.py) and implemented by [nli_labeler.py](../src/tree/nli_labeler.py).

---

## 4. Paper method: Hidden Markov Tree and belief propagation

BTProp treats truth as hidden and LLM confidence as observed evidence. For each node \(u\), the two central distributions are:

### 4.1 Emission probabilities

\[
P(S_u\mid Z_u)
\]

This is the probability of observing an LLM confidence score, given the true state of the statement. The score is not assumed to be a calibrated posterior probability.

The paper estimates emissions from labeled data. It bins scores into:

\[
[0,0.2),\ [0.2,0.4),\ [0.4,0.7),\ [0.7,0.9),\ [0.9,1.0].
\]

Its reported table is:

| Truth state | \([0,.2)\) | \([.2,.4)\) | \([.4,.7)\) | \([.7,.9)\) | \([.9,1]\) |
|---|---:|---:|---:|---:|---:|
| \(Z=T\) | 0.12 | 0.05 | 0.10 | 0.08 | 0.65 |
| \(Z=F\) | 0.30 | 0.10 | 0.15 | 0.13 | 0.32 |

Each row is a conditional distribution over confidence bins for one true state. For example, the final entry in the first row means that, in the paper's calibration sample, 65% of true statements fell in the highest confidence bucket.

### 4.2 Transition probabilities

For a parent \(u\), the transition distribution is:

\[
P(Z_{C(u)}\mid Z_u).
\]

It says how the children's hidden truth values depend on the parent's truth value.

For decomposition children, the paper uses the joint relationship:

\[
P(Z_{C(u)}=\mathbf z\mid Z_u=T)
=
\begin{cases}
1 & \mathbf z=(T,\ldots,T),\\
0 & \text{otherwise}.
\end{cases}
\]

When \(Z_u=F\), the all-true assignment has probability zero, and the remaining assignments are distributed uniformly:

\[
P(Z_{C(u)}=\mathbf z\mid Z_u=F)
=
\begin{cases}
0 & \mathbf z=(T,\ldots,T),\\
\frac{1}{2^{m}-1} & \text{otherwise},
\end{cases}
\]

where \(m=|C(u)|\).

For premises and corrections, each child has its own pairwise conditional transition, and the children are conditionally independent given the parent. The transition values depend on the relationship type. In the paper, unconstrained probabilities \(p_t\) and \(p_f\) are both set to \(0.5\).

The repository's exact values are in [relationships.py](../src/tree/relationships.py). With the repository's `label(child, parent)` direction, they are:

| Relationship | \(P(Z_c=T\mid Z_p=T)\) | \(P(Z_c=T\mid Z_p=F)\) | Meaning under repository direction |
|---|---:|---:|---|
| Equivalence | 1.0 | 0.0 | Same truth value |
| Entailment | 0.5 | 0.0 | Child implies parent; a false parent rules out a true child |
| Reverse entailment | 1.0 | 0.5 | Parent implies child; a true parent forces a true child |
| Contradiction | 0.0 | 0.5 | A true parent rules out a true contradictory child |

The \(0.5\) entries represent a truth value that is not determined by that direction of implication. The transition values are logical modeling assumptions, not parameters fitted by this project.

### 4.3 Upward belief propagation

For a node \(u\), let \(S_{T(u)}\) denote all observed confidence scores in the subtree rooted at \(u\). Define:

\[
\beta(z,u)=P(S_{T(u)}\mid Z_u=z).
\]

For a leaf:

\[
\beta(z,u)=P(S_u\mid Z_u=z).
\]

For an internal node with \(m\) children:

\[
\beta(z,u)
=P(S_u\mid Z_u=z)
\sum_{\mathbf z\in\{T,F\}^{m}}
P(Z_{C(u)}=\mathbf z\mid Z_u=z)
\prod_{v\in C(u)}\beta(z_v,v).
\]

The computation starts at leaves and moves toward the root. It sums over every possible truth assignment of the children, multiplying the transition probability by the likelihood of the evidence in each child subtree.

The root posterior is:

\[
P(Z_0=T\mid S_{T(0)})
=
\frac{\beta(T,0)P(Z_0=T)}
{\sum_{z\in\{T,F\}}\beta(z,0)P(Z_0=z)}.
\]

The paper uses \(P(Z_0=T)=P(Z_0=F)=0.5\), so this simplifies to:

\[
P(Z_0=T\mid S_{T(0)})
=\frac{\beta(T,0)}{\beta(T,0)+\beta(F,0)}.
\]

The recurrence is implemented in [compute_beta](../src/inference/hmt.py); [posterior_root_true](../src/inference/hmt.py) applies the root prior and normalizes.

### 4.4 Worked numerical example

The following is illustrative, not a reported paper result. It uses the paper's highest-bin emission values and the repository's contradiction transition.

Suppose the root confidence is \(0.95\), in the highest bin:

\[
P(S_r\mid Z_r=T)=0.65,\qquad P(S_r\mid Z_r=F)=0.32.
\]

With a \(0.5\) prior and no children:

\[
P(Z_r=T\mid S_r)
=\frac{0.5(0.65)}{0.5(0.65)+0.5(0.32)}
\approx 0.670.
\]

Now add a correction child whose observed score is \(1.0\). Its special emissions are \(0.8\) if true and \(0.2\) if false. The child contradicts the root, so:

\[
P(Z_c=T\mid Z_r=T)=0,\qquad P(Z_c=T\mid Z_r=F)=0.5.
\]

If the root is true, the child must be false, giving child evidence likelihood \(0.2\). If the root is false, the child has equal probability of being true or false, giving likelihood \(0.5(0.8)+0.5(0.2)=0.5\). The unnormalized root weights are therefore:

\[
w_T=0.5(0.65)(0.2)=0.065,\qquad
w_F=0.5(0.32)(0.5)=0.080.
\]

Thus:

\[
P(Z_r=T\mid S_r,S_c)
=\frac{0.065}{0.065+0.080}\approx0.448.
\]

The contradiction-linked evidence shifts the root posterior from approximately \(0.670\) to \(0.448\). This illustrates the inference mechanism; it does not imply that generated corrections are always valid or that the model always improves an individual decision.

---

## 5. Datasets used by the paper

The paper evaluates on three datasets:

| Dataset | Content | Evaluation unit |
|---|---|---|
| WikiBio-GPT3 | LLM-generated biographies | Sentence |
| FELM-Science | ChatGPT answers to science questions, annotated by segment | Segment |
| FactCheckGPT | Fine-grained factual claims from LLM outputs | Sentence/claim |

The datasets differ in topic and construction. WikiBio-GPT3 is biography-heavy; FELM-Science and FactCheckGPT cover broader topics, including scientific knowledge.

### 5.1 Paper preprocessing

The paper describes preprocessing WikiBio-GPT3 and FELM-Science in two stages:

1. **Decontextualization:** rewrite context-dependent statements so pronouns and references can be understood without the surrounding answer. The paper says it used GPT-3.5-turbo-0125 for this.
2. **Manual filtering:** remove non-checkable filler statements that the benchmark nevertheless labeled true, such as “Sure!” or generic closings.

The paper reports that its preprocessed files were supplied in supplementary material and would be made public. Those manually prepared files were not available to this repository in the same form.

FactCheckGPT claims were already manually processed to contain individual claims, so the paper did not apply decomposition to those statements.

### 5.2 Paper's emission-calibration split

The paper uses the first 120 WikiBio-GPT3 examples—about half the dataset—to estimate its emission probabilities. It uses GPT-3.5-turbo-0125 to obtain confidence scores and labels for the calibration table, then applies those estimated emissions to:

- the remaining WikiBio-GPT3 examples,
- FELM-Science,
- FactCheckGPT.

The paper also uses the same estimated emissions for Llama-3-8B-Instruct. Thus, the paper's reported calibration is not a separate fit on this repository's FELM-Science split.

---

## 6. Paper experiments and reported results

The paper compares BTProp with prior confidence, chain-of-thought prompting, semantic uncertainty, SelfCheckGPT, and FOCUS where applicable. It evaluates GPT-3.5-turbo-0125 and Llama-3-8B-Instruct.

On FELM-Science, Table 3 reports:

| Backbone | Method | AUROC | AUC-PR | F1 | Accuracy |
|---|---|---:|---:|---:|---:|
| GPT-3.5-turbo-0125 | Prior Confidence | 75.5 | 37.2 | 42.1 | 81.6 |
| GPT-3.5-turbo-0125 | BTProp | 79.1 | 52.3 | 56.5 | 81.6 |
| Llama-3-8B-Instruct | Prior Confidence | 76.1 | 38.2 | 44.0 | 80.6 |
| Llama-3-8B-Instruct | BTProp | 77.8 | 48.2 | 51.3 | 82.8 |

The table reports metrics as percentages: for example, 77.8 is AUROC \(0.778\). The paper reports a maximum belief-tree depth of 2; tree construction is greedy apart from correction generation, which samples five answers at temperature \(0.7\).

The paper's results vary by dataset and metric. Its headline is not that BTProp wins every comparison: for example, SelfCheckGPT performs better on WikiBio-GPT3. The reported strong results are especially on FELM-Science and FactCheckGPT.

---

## 7. What this repository reproduces

This repository independently implements the method rather than directly running the authors' code. It follows the main methodological structure:

\[
\text{generate related claims}
\rightarrow
\text{label relationships}
\rightarrow
\text{score claims}
\rightarrow
\text{calibrate scores}
\rightarrow
\text{propagate evidence}
\rightarrow
P(\text{root true}).
\]

The repository adopts:

- the three generation strategies;
- LLM strategy selection for premises/corrections;
- the official BTProp prompts, reproduced with attribution;
- bidirectional NLI relationship labeling;
- the relationship-specific transition model;
- correction nodes with score \(1.0\) and special \(0.8/0.2\) emission values;
- a root prior of \(0.5\);
- True/False token probabilities for its main open-model confidence path;
- FELM-Science as its reproduction dataset.

The relevant implementation is:

- [strategies.py](../src/tree/strategies.py) — generation strategy parsing and filtering;
- [builder.py](../src/tree/builder.py) — breadth-first construction of the raw tree;
- [nli_labeler.py](../src/tree/nli_labeler.py) — directional NLI combination;
- [hmt.py](../src/inference/hmt.py) — exact tree inference;
- [pipeline.py](../src/pipeline.py) — converts the raw tree and wires verification;
- [hf_client.py](../src/llm/hf_client.py) — local token-probability confidence.

For the main open-model run, the project uses `NousResearch/Meta-Llama-3-8B-Instruct`; its NLI checkpoint is `cross-encoder/nli-deberta-v3-large`.

---

## 8. What differs from the base paper

The project is a method reproduction, not a controlled exact replication. These differences matter when interpreting comparisons.

| Aspect | Base paper | This repository |
|---|---|---|
| Maximum tree depth | 2 | 1, to control LLM-call cost |
| Emission calibration data | First 120 WikiBio-GPT3 examples | FELM-Science fit half |
| FELM preprocessing | Paper's manually cleaned, stand-alone data | Public raw data plus local cleaning and stand-alone rewriting |
| Stand-alone rewriting | GPT-3.5-turbo-0125 as preprocessing | Llama rewriting with an NLI entailment guard |
| Emission model | Five-bin table | Five-bin table plus an additional logistic calibration |
| Fit/test protocol | Paper's dataset-specific setup | Whole-answer, stratified FELM split with seed 42 |
| F1 threshold | Paper's evaluation procedure | Cross-fitted threshold by answer group |
| Scope | Per-statement belief trees | Per-statement trees plus answer-level graph and surrogate extensions |

Other implementation deviations are recorded in decision D29 of [03-decisions.md](./project-log/03-decisions.md), including parsing fixes, multiple-sample caching, and retaining a root confidence when decomposition returns no check-worthy claim.

---

## 9. Dataset used by this repository: FELM-Science

The project downloads the raw `all.jsonl` from the Hugging Face dataset `hkust-nlp/felm` and keeps the science domain. FELM consists of ChatGPT answers split into segments, with human true/false labels. The loader and cleaning/splitting code are in [fit_emission_model.py](../scripts/fit_emission_model.py).

The current README reports:

| Split | Answers | Segments | False |
|---|---:|---:|---:|
| Fit | 62 | 291 | 53 |
| Test | 61 | 304 | 49 |
| Total | 123 | 595 | 102 |

The fit split is used for calibration and fit-stage choices. The test split is held out for final reported evaluation. Answers, rather than individual segments, are split so related sentences from one answer cannot appear in both sets. The split is stratified by whether an answer contains any false segment and uses seed 42.

### 9.1 Cleaning

The repository filters out:

- duplicate answers;
- very short, non-claim segments (the code uses a four-word minimum);
- list openers that are not claims, such as an introductory phrase ending in a list number.

The thresholds and split logic are defined in [fit_emission_model.py](../scripts/fit_emission_model.py), notably `is_claim`, `split_answers`, and `load_felm_science_answers`.

### 9.2 Stand-alone rewriting

Raw segments can depend on context—for example, “They are located on the wings.” The paper's manually rewritten data was unavailable, so this project approximates that preprocessing:

1. Provide the original question, full answer, and segment to the LLM.
2. Ask it to replace pronouns/references and add only needed context, preserving the meaning and any errors.
3. Keep a proposed rewrite only if NLI says the rewrite entails the original.
4. Otherwise retain the original segment.
5. Keep the human label attached to the original segment.

The code is in [decontextualize_felm.py](../scripts/decontextualize_felm.py). The entailment guard is conservative: it is designed to reject a rewrite that drops potentially false content, but it may also reject a faithful rewrite.

There is a small count discrepancy in the current documentation: the README's cleaned-data table totals 595 segments, while the rewrite summary refers to 594. The project log also contains historical counts from prior preprocessing iterations. When exact counts matter, consult the current code and result JSON rather than treating older prose counts as interchangeable.

---

## 10. The repository's calibration models

The repository fits the calibration on the FELM fit split in [fit_emission_model.py](../scripts/fit_emission_model.py).

### 10.1 Five-bin lookup emission

The repository uses bin edges \(0.19,0.39,0.69,0.89\), giving five buckets. For each truth state \(z\), it counts the fraction of fit-set scores in each bucket:

\[
\widehat P(B=b\mid Z=z)
=\frac{\#\{\text{fit examples with }Z=z\text{ and score in }b\}}
{\#\{\text{fit examples with }Z=z\}}.
\]

The fitted bucket is then used as the emission likelihood for a node's score. This is conceptually the paper's histogram calibration, but the training dataset differs.

### 10.2 Logistic calibration

The repository also fits a logistic regression of the human true/false label on the logit of confidence:

\[
\operatorname{logit}P(Z=T\mid s)
=a\operatorname{logit}(s)+b,
\]

where:

\[
\operatorname{logit}(s)=\log\frac{s}{1-s}.
\]

Scores are clipped away from 0 and 1 before taking the logit. The fitted posterior is converted into an emission likelihood using the fitted true-label base rate \(\pi=P(Z=T)\):

\[
P(S=s\mid Z=T)=\frac{P(Z=T\mid S=s)}{\pi},
\qquad
P(S=s\mid Z=F)=\frac{1-P(Z=T\mid S=s)}{1-\pi}.
\]

The common score-density term cancels in the truth posterior. The code for this conversion is `LogisticEmission` in [hmt.py](../src/inference/hmt.py); fitting is `fit_logistic` in [fit_emission_model.py](../scripts/fit_emission_model.py).

The logistic curve was added because the small FELM fit set made some histogram buckets unstable. It is an extra project variant, not the paper's original emission table.

---

## 11. The repository's reported results and their scope

The current README reports results on 304 held-out test sentences, 49 labeled false:

| Method | Calibration | AUROC | AUC-PR | F1 |
|---|---|---:|---:|---:|
| Raw Llama confidence | — | 0.745 | 0.329 | 0.371 |
| BTProp-style tree | Five-bin table | 0.678 | 0.362 | 0.393 |
| BTProp-style tree | Logistic curve | 0.741 | 0.403 | 0.439 |
| Answer graph, no contradiction links | Logistic curve | 0.773 | 0.475 | 0.438 |

These are the repository's results, not the paper's. The paper's Llama FELM-Science BTProp result is AUROC \(0.778\), AUC-PR \(0.482\), and F1 \(0.513\); its Prior Confidence result is \(0.761\), \(0.382\), and \(0.440\). The repository's test set, preprocessing, depth, calibration, and F1 threshold differ, so the closeness of some numbers should not be interpreted as a controlled win or loss.

The repo's headline tree result is nearly unchanged AUROC versus raw confidence, but it improves AUC-PR. The answer graph improves ranking metrics under the chosen link rule. The small test set means uncertainty is substantial; the project's README reports answer-cluster bootstrap intervals and cautions against overclaiming significance.

---

## 12. What “replication” means in this project

The repository replicates the **methodological chain**, not every experimental condition:

\[
\text{LLM-generated tree}
\rightarrow
\text{bidirectional NLI labels}
\rightarrow
\text{emission and transition probabilities}
\rightarrow
\text{exact tree belief propagation}
\rightarrow
P(\text{root true}).
\]

It reproduces the official prompts and core BTProp assumptions in its own code. It differs in the calibration source, data preprocessing, tree depth, split protocol, and thresholding. It also adds methods beyond the paper.

So the accurate description is:

> Reason-Graph is an independent implementation of the BTProp method, evaluated on a locally cleaned and split FELM-Science dataset, with a shallower tree and additional answer-graph and surrogate experiments.

It is not accurate to say that every paper setting was reproduced exactly.

---

## 13. What comes next in the code-level study

The key conceptual separation to keep in mind is:

\[
\underbrace{P(S_u\mid Z_u)}_{\text{how confidence behaves for true/false claims}}
\qquad\text{and}\qquad
\underbrace{P(Z_{C(u)}\mid Z_u)}_{\text{how claim relationships constrain truth}}.
\]

The first comes from calibration data. The second comes from the relationship-specific logical model. Exact tree inference combines them to obtain the root posterior.

The next code-level step is to trace those components in [hmt.py](../src/inference/hmt.py), then follow how [builder.py](../src/tree/builder.py) constructs a tree and how [pipeline.py](../src/pipeline.py) connects the tree to inference.
