# Running the Reproduction on Kaggle

**Why Kaggle:** the paper's confidence method needs the model's internal True/False
probabilities. Free APIs don't give them, and the model (Llama-3-8B-Instruct, 16 GB) doesn't fit
reliably on a 6 GB laptop GPU. Kaggle gives free GPUs (2 × 16 GB T4, ~30 h/week), and runs keep
going after you close the laptop.

**What runs:** `notebooks/kaggle_felm_llama3.ipynb` — downloads our code from GitHub, rewrites
every sentence to stand alone (D30), builds the calibration on the fit half, evaluates on the
test half with the paper's own tree-building method (D29), and packages the results.

## One-time setup

### 1. Kaggle account
1. Sign up at kaggle.com.
2. *Settings → Phone verification.* Required for GPU and internet access.

### 2. GitHub token (so Kaggle can read our private repo)
1. GitHub → *Settings → Developer settings → Personal access tokens → Fine-grained tokens →
   Generate new token*.
2. Name: `kaggle-reason-graph`. Expiration: 30 days.
3. *Repository access → Only select repositories →* `Reason-Graph`.
4. *Permissions → Repository permissions → Contents: Read-only.* Nothing else.
5. Generate, copy the token (starts with `github_pat_`).

It can only **read** this one repo, and it expires. Never paste it into a notebook cell or chat.

### 3. Put the token in Kaggle Secrets
1. Kaggle → *Create → New Notebook*.
2. *File → Import Notebook →* upload `notebooks/kaggle_felm_llama3.ipynb` from the repo.
3. Right panel → *Add-ons → Secrets → Add Secret*. Label: `GITHUB_TOKEN`, value: the token.
   Tick the box to attach it to this notebook.

## Each run

1. **Push the branch first.** Kaggle downloads whatever is on GitHub, not your laptop:
   `git push origin phase-2/tree-construction`
2. In the notebook's right panel:
   - *Session options → Accelerator →* **GPU T4 x2**
   - *Session options → Internet →* **On**
3. *(Optional, recommended once)* Quick check before the long run: in an interactive session,
   run cells 1–5, then run
   `!python -u scripts/evaluate_felm.py --backend hf --model {MODEL} --limit 3` — confirms
   everything loads, in a few minutes.
4. **Save Version → Save & Run All (Commit).** This runs on Kaggle's servers. You can close the
   tab and the laptop.
5. When it finishes (~5–7 h expected), open the version → **Output** tab → download
   `reason_graph_results.zip`.
6. Unzip it into the repo root on your laptop. You get `results/*.json` (numbers + per-statement
   scores) and `llm_cache/` (every model answer, so re-analysis needs no GPU).

## If something goes wrong

| Symptom | Cause / fix |
|---|---|
| `kaggle_secrets` error / 401 on clone | Secret not attached to this notebook, or token expired |
| `Could not resolve host` | Internet is off, or phone not verified |
| CUDA out of memory | Accelerator is P100 or single T4 — choose **GPU T4 x2** |
| Run stopped at 12 h | Kaggle's limit. Re-run: everything cached in that version's output is not carried over automatically — upload the previous `reason_graph_results.zip` as a Kaggle Dataset and unzip it into the repo before running, then it resumes |

## Cost of the run (all free)

- Stand-alone rewrites: 595 short generations (~40 min).
- Fit half: 291 forward passes (minutes).
- Test half: 304 belief trees. Each: decomposition, strategy choice, then premises (2 samples) and/or
  corrections (1 question, 5 sampled answers, 5 revisions), plus confidences and the NLI model.
  Several hours on 2 × T4; samples run in parallel on the GPU.
- Kaggle stops a session at 12 h. If that happens, see the table above.

## What to check in the output

- Step 5 prints how many segments were rewritten, how many rewrites the safety check rejected, and
  15 examples. Skim them: each rewrite should say the same thing as the original, only clearer.
