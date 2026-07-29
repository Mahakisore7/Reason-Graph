# Research Notes — BTProp (the base paper)

Condensed reference for explaining the base paper to your mentor without re-reading the whole
paper each time. For the full first-principles build-up (probability → graphical models →
belief propagation), see the companion document *"BP-Verify: Complete Project Guide"* — this file
only covers the base paper itself.

## Citation

**Title:** A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation (BTProp)
**Authors:** Bairu Hou, Yang Zhang, Jacob Andreas, Shiyu Chang
**Venue:** NAACL 2025 (Main Conference)
**Links:** [arXiv](https://arxiv.org/abs/2406.06950) · [ACL Anthology](https://aclanthology.org/2025.naacl-long.158/) · [Official code](https://github.com/UCSB-NLP-Chang/BTProp)

## The One-Paragraph Version

Given a claim, build a tree of related child claims using three different LLM-prompted strategies.
Ask the LLM for its own confidence on every node. Model the tree as a Hidden Markov Tree — hidden
truth value per node, observed confidence per node — and run the standard exact upward-downward
(belief propagation) algorithm to compute a corrected posterior for the root claim.

## Why It Exists

Two gaps in prior consistency-checking methods: (1) their augmented statements are repetitive and
unstructured rather than logically diverse, and (2) they collapse continuous confidence into hard
true/false before reasoning about it, throwing away exactly the signal that separates a shaky guess
from a firm judgement.

## The Method, Step by Step

1. **Build the tree.** Three child-generation strategies:
   - **Decomposition** — split a multi-fact statement into sub-facts (jointly equivalent to the
     parent).
   - **Supportive & Contradictory Premises** — LLM generates premises for/against the parent.
   - **Statement Correction** — LLM generates what it believes the corrected statement is (3-step:
     question → sampled answers → corrected statement per answer); these children get confidence
     fixed at 1.0, not re-probed.
   - Decomposition is tried first; if it yields only one statement, the LLM picks between the other
     two strategies.
2. **Label parent-child relationships.** Four types — equivalence, entailment, reverse entailment,
   contradiction — via an NLI model run in both directions (decomposition's relationship is fixed
   by construction instead).
3. **Score confidence.** Prompt `"True or False? {statement}"`, normalise the LLM's own next-token
   probabilities for "True"/"False" into `S_u ∈ [0,1]`.
4. **The Hidden Markov Tree.** Per node `u`: hidden `Z_u ∈ {T,F}`, observed `S_u`.
   - Emission `p(S_u | Z_u)` — estimated empirically by binning `[0,1]` into 5 ranges on a 120-example held-out labelled set.
   - Transition `p(Z_children | Z_parent)` — near-deterministic for decomposition; a small
     4-relationship-type table (with parameters `p_t = p_f = 0.5`) for premises/corrections.
   - Prior `p(Z_u)` — uniform 0.5/0.5.
5. **Inference.** The upward-downward algorithm. Define `β(z, u)` = probability of observing every
   score in `u`'s sub-tree given `Z_u = z`. Leaves: `β(z,u) = p(S_u | Z_u=z)`. Internal nodes:
   multiply the node's own emission by a sum, over every joint child truth-setting, of the
   transition probability times the product of children's `β` values. Root posterior is the
   normalised ratio of `β(T, root)` to `β(F, root)`. **The paper's own text calls this a belief
   propagation process.**

## Experimental Setup

| Item | Value |
|---|---|
| Datasets | Wikibio-GPT3, FELM-Science, FactCheckGPT |
| Backbones | GPT-3.5-turbo-0125, Llama-3-8B-Instruct |
| Baselines | Prior Confidence, CoT, SelfCheckGPT, Semantic Uncertainty, Maieutic Prompting, Focus |
| Metrics | AUROC, AUC-PR, F1, Accuracy |
| Max tree depth used | 2 |

## Results — Reported Honestly

- Best AUROC/AUC-PR on **FELM-Science and FactCheckGPT** (both backbones), beating the best
  baseline by roughly 3–9%.
- **Exception:** on Wikibio-GPT3, SelfCheckGPT actually wins — BTProp is competitive but not best
  there. Say this out loud in any review; don't let the abstract's "improves over baselines" framing
  overstate it.
- Ablation: any single strategy alone underperforms the full combination; decomposition-only or
  premise-only alone can even underperform just asking for prior confidence directly. Correction
  alone is the strongest single strategy, but still worse than all three combined.
- Efficiency: baselines plateau with more samples; BTProp keeps trading tree depth for performance,
  though it costs more time per prediction in absolute terms.

## Limitations — Author-Acknowledged

High time cost: many LLM queries per tree, growing exponentially with depth. The authors'
own suggested (unimplemented) fix: selectively expand only the most useful nodes. **This is
precisely what our Extension 2 (surrogate scorer) answers.**

## Limitations — Additional, Identified By Us

- Tree-only: no node can have more than one parent, so a claim depending on two+ earlier claims
  can't be represented. **This is precisely what our Extension 1 (DAG + loopy BP) answers.**
- Only tested at depth 2.
- Evaluated on standalone factual statements, not literal multi-step chain-of-thought traces.
- English-only, general-knowledge domain.

## One Line for Any Review

*"BTProp verifies a single claim by building a tree of related claims and running exact belief
propagation over a Hidden Markov Tree; we generalise that to a general reasoning graph with loopy
BP, and make it cheaper with a surrogate confidence model — both directions the original authors
either couldn't do or explicitly flagged as future work."*
