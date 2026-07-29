# Architecture

## System Overview

The pipeline turns one target statement (or one full reasoning chain) into a corrected,
logically-consistent probability that it's trustworthy. Blue = base paper (BTProp); this is our
system boundary either way, extended where marked.

```mermaid
flowchart TB
    A["Target statement / reasoning chain"] --> B["Graph Construction<br/>(tree — BTProp; DAG — ours)"]
    B --> C["Confidence Scoring<br/>LLM: 'True or False?' -> S_u"]
    C --> D["Probabilistic Model<br/>hidden Z_u, observed S_u,<br/>emission + transition probs"]
    D --> E["Inference<br/>tree: upward-downward (exact)<br/>DAG: loopy BP (iterative)"]
    E --> F["Posterior P(Z_root = True)"]
    F --> G["Decision: trustworthy / likely hallucination"]

    X["Our Ext. 1: Surrogate Scorer"] -.skips some LLM calls in.-> C
    Y["Our Ext. 2: DAG dependency extraction"] -.replaces tree-only step in.-> B
```

## Module Boundaries

Each stage above maps to one Python package under `src/bpverify/`, kept independent so any one
can be unit-tested and swapped without touching the others.

```mermaid
flowchart LR
    subgraph llm["bpverify.llm"]
        L1["LLMClient (protocol)"]
        L2["OpenAIClient / LocalHFClient"]
        L3["Prompt templates"]
    end
    subgraph tree["bpverify.tree"]
        T1["Graph construction<br/>(3 strategies)"]
        T2["NLI-based relationship labelling"]
    end
    subgraph inference["bpverify.inference"]
        I1["hmt.py — exact tree BP<br/>(implemented, Phase 1)"]
        I2["loopy.py — general-graph BP<br/>(Phase 3)"]
    end
    subgraph surrogate["bpverify.surrogate"]
        S1["Confidence predictor<br/>(Phase 4)"]
    end
    subgraph eval["bpverify.eval"]
        E1["Metrics: AUROC / AUC-PR / F1"]
        E2["Benchmark loaders"]
    end

    tree --> llm
    tree --> inference
    surrogate -.can substitute for.-> llm
    inference --> eval
```

**Design rule:** `bpverify.inference` never imports `bpverify.llm` directly — it only consumes
plain data (a graph of nodes with observed scores). This is what makes `hmt.py` fully unit-testable
with hand-constructed numbers and no network access, no API key, and no mocking required (see
`tests/test_hmt.py`).

## Data Flow (single verification request)

```mermaid
sequenceDiagram
    participant U as Caller
    participant T as tree/graph builder
    participant L as LLM client
    participant M as Probabilistic model
    participant I as Inference engine

    U->>T: build_graph(statement, max_depth)
    T->>L: generate children (3 strategies)
    L-->>T: candidate child statements
    T->>L: score each node ("True or False?")
    L-->>T: confidence scores S_u
    T->>M: assemble HiddenMarkovTree(nodes, edges, S_u)
    M->>I: run inference
    I-->>U: posterior P(Z_root = True), per-node beliefs
```

## Why a Tree Structure Is Implemented First (Phase 1), Before the DAG (Phase 3)

The exact upward-downward recursion only works because a tree guarantees each node has exactly
one parent — this is what makes the base algorithm both correct-by-construction and the simplest
possible thing to get right first. Building `hmt.py` (tree case) before `loopy.py` (general DAG) is
deliberate risk sequencing: prove the core math against hand-computed examples on the easy case
before adding the part (cyclic message passing, convergence checking) that has no formal guarantee.

## Suggested Repository Layout

```
bp-verify/
├── docs/                    # this folder — planning + architecture + research notes
├── src/bpverify/
│   ├── tree/                # Phase 2: graph construction (3 strategies) + NLI labelling
│   ├── llm/                 # Phase 1: LLM client protocol + implementations + prompts
│   ├── inference/
│   │   ├── hmt.py           # Phase 0: exact upward-downward algorithm on a tree
│   │   └── loopy.py         # Phase 3: general-graph loopy BP
│   ├── surrogate/           # Phase 4: cheap confidence predictor
│   ├── eval/                # Phase 2/7: metrics + benchmark loading
│   └── pipeline.py          # Phase 5: end-to-end wiring
├── tests/                   # pytest — one test module per src package, written by you
├── scripts/                 # small CLI entry points (reproduce baseline, run eval, etc.)
├── notebooks/               # exploratory analysis, created as needed (gitignored outputs)
└── data/                    # gitignored — local copies of FELM-Science, FactCheckGPT, etc.
```
