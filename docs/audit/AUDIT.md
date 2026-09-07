# MBBN code audit

Audit of [`Transconnectome/MBBN`](https://github.com/Transconnectome/MBBN) at commit
`eed038c` ("Communications Biology ver."), the code accompanying *Learning brain
dynamics across distinct scaling regimes reveals psychiatric signatures*
(Commun Biol 2026, [10.1038/s42003-026-10011-7](https://doi.org/10.1038/s42003-026-10011-7)).

**35 findings** across 15 Python files / 3,461 lines. **15** were established by running the repository's own modules rather than
by reading them; the rest are code-reading findings whose mechanism is stated in full so
a reader can check them independently.

| severity | count | meaning |
|---|---|---|
| blocking | 7 | Blocking — the released code cannot run |
| high | 9 | High — changes reported numbers or their interpretation |
| medium | 14 | Medium — degrades training, robustness or cost |
| low | 5 | Low — hygiene |

| area | count |
|---|---|
| Correctness | 7 |
| Evaluation protocol | 5 |
| Training safety | 6 |
| Robustness | 9 |
| Interpretability | 4 |
| Cost / performance | 4 |

---

## Method, and what this audit does not establish

Every statement below was checked against the code at that commit. Measured findings were
obtained by importing the released modules unmodified and exercising them; where the
released code would not import on a current dependency stack, the specific incompatibility
is itself reported as a finding (D01, D02, D03) and was worked around only to reach the
next layer.

**None of ABCD, UK Biobank or ABIDE is available to this audit.** No claim here is a claim
about the paper's reported accuracy. Specifically:

- No fix was validated against a published metric, and none of the changes is claimed to
  improve AUROC. Where a change should help, the reasoning is stated as a mechanism, not
  as a measured improvement.
- Quantities that depend on cohort size (how many subjects a dropped batch discards, how
  much the band cache saves) are computed from the cohort sizes this repository's own
  README tabulates — UKB 40,699, ABCD 8,833, ABIDE 141, summing to the 49,673 the paper
  states — paired with the epoch counts in the launch scripts. Those N are the released
  documentation's figures, not counts verified against the cohorts themselves, and the
  split sizes follow from applying the released `train_test_split` calls to them.
- Measurements on synthetic data (D12, D13, D17, D29) demonstrate the *mechanism* and its
  magnitude under stated conditions. They are not estimates of the effect on the paper's
  numbers.

## Three findings that bear on published claims

### 1. The interpreted attention maps carry no label information (D08, D09)

`Transformer_Finetune_Three_Channels` computes three band-specific ROI x ROI spatial
attention maps and returns them alongside the prediction, but they never enter the
prediction. Backpropagating the prediction gives

```
d(prediction) / d(high_spatial_attention.qkv.weight)     = 0.0   (exactly)
d(prediction) / d(low_spatial_attention.qkv.weight)      = 0.0
d(prediction) / d(ultralow_spatial_attention.qkv.weight) = 0.0
```

while the band-repulsion loss delivers |grad| ~2817 to the same parameters. The maps are
therefore shaped *only* by the repulsion objective. Two consequences:

- The frequency-resolved connectivity patterns reported as ADHD and ASD signatures are
  functions of the repulsion objective and the data, never of the diagnosis.
- `visualization.py` computes its saliency by backpropagating that same repulsion loss.
  For `S` a sum of mean-L1 distances, `d(-log S)/dh_ij = -[sign(h-l) + sign(h-u)] /
  (numel * S)`, which takes only the five values `{-2,-1,0,1,2}` times one global scalar.
  Measured on a (2, 8, 400, 400) map: **4 distinct values across 2.56M entries, 33.4%
  exactly 0**. Gradient magnitude carries no per-edge information, and the class label
  enters only through which subjects are selected for averaging.

The step-1 `vanilla_BERT` baseline *does* wire its spatial branch into the prediction
(|grad| 406). A step-1-versus-step-2 comparison therefore changes the head wiring as well
as the frequency decomposition, and is not an isolated test of multi-band modelling.

### 2. Reported operating-point metrics are prevalence-determined constants (D12)

`LossWriter` is constructed with `Trainer.val_threshold`, which is `0`, and nothing
updates it afterwards. `Metrics.ROC_CURVE(name='test')` therefore thresholds *sigmoid*
outputs at 0, so every subject is predicted positive. Reproduced through the released
`Metrics` on synthetic scores with AUROC 0.933 and prevalence 0.275:

| | threshold | sensitivity | specificity | balanced acc. | F1 |
|---|---|---|---|---|---|
| validation (threshold chosen on val) | 0.506 | 0.900 | 0.866 | 0.878 | 0.794 |
| **test, as released** | **0.000** | **1.000** | **0.000** | **0.500** | **0.431** |
| test, validation threshold carried over | 0.506 | 0.891 | 0.866 | 0.878 | 0.794 |

At threshold 0, sensitivity is 1 and specificity 0 by construction, balanced accuracy is
pinned at 0.500, and F1 equals `2p/(1+p)` = 0.431 at this prevalence — all independent of
the model. AUROC is unaffected, since it does not use a threshold.

### 3. The band-repulsion objective is optimised by hub attention (D17)

The loss is `-log S` with `S = L1(h,l) + L1(h,u) + L1(l,u)`, each term a *mean* absolute
difference between row-stochastic attention maps. Two properties:

- Rows are probability vectors, so each pairwise mean-L1 is at most `2/N` and
  `S <= 6/N`. At N=400 that ceiling is 0.015, and it is **attained** — measured
  0.0150 — exactly when every row is one-hot and the three bands point at different
  targets. The global optimum of the objective is single-target (star / hub) attention,
  independently of the data.
- `S = 0` whenever the three maps coincide exactly, so `-log S` is singular there and its
  gradient scales as `1/S`. Random initialisation is not that point but is close to it:
  measured `S = 0.00278` at the released initialisation, giving a finite loss of 588.6 at
  lambda=100, and the gradient grows as the scale of the attention logits shrinks —
  measured infinity-norm 0.50 at logit scale 1e-4 versus 1.6e-3 at scale 1.

400 steps of the repulsion term alone, at the released fine-tuning learning rate on the
released `Attention` module, moved mean row entropy from 0.9908 to 0.9625 of `log N` and
the mean peak row weight from 2.55/N to 6.11/N — a 2.39x increase, monotonic across all
sampled steps.

This matters for interpretation: a prior analysis of this paper's supplementary tables
found that its top-100 features form a single-hub star structure per band. An objective
whose optimum is one-hot attention is a mechanism that would produce exactly that,
without any corresponding property of the data.

---

## All findings

### **Blocking** — the released code cannot run (7)

#### D01 · Correctness · `model.py:173`

*Released:* init_weights() is called before self.bert and self.cls_embedding are assigned, so it walks an empty module tree; it is also not the documented entry point.

*Consequence:* Nothing is initialised, and on transformers>=5 construction raises AttributeError: no attribute 'all_tied_weights_keys'. The released model does not instantiate on a current release at all.

*Fix:* post_init() after the submodules exist.

*Evidence:* measured: construction traceback reproduced

#### D02 · Correctness · `dataloaders.py:376`

*Released:* np.in1d, removed in NumPy 2.0.

*Consequence:* AttributeError at split time on any NumPy>=2 install.

*Fix:* np.isin.

*Evidence:* measured: AttributeError reproduced

#### D03 · Correctness · `trainer.py (3 sites)`

*Released:* torch.cuda.nvtx.range_push/pop called unconditionally.

*Consequence:* RuntimeError 'NVTX functions not installed' on any CPU-only build, so the pipeline cannot be smoke-tested without a GPU.

*Fix:* Guarded _nvtx_push/_nvtx_pop helpers.

*Evidence:* measured: RuntimeError reproduced

#### D04 · Correctness · `model.py:19`

*Released:* Attention.forward references self.proj and self.proj_drop, which __init__ never creates.

*Consequence:* AttributeError whenever the feature path (rather than the attention matrix) is requested.

*Fix:* Output projection created; --attn_only skips the unused value path entirely. Flag: `--attn_only`.

*Evidence:* code-read + test

#### D05 · Correctness · `model.py:213`

*Released:* num_heads chosen by 'if seq_len%12==0 elif seq_len%8==0' with no else.

*Consequence:* UnboundLocalError for any sequence length divisible by neither (e.g. 280 -> ABIDE at its native length).

*Fix:* pick_spatial_heads() with a divisor fallback.

*Evidence:* measured: test pins 280/348/464/97

#### D06 · Correctness · `learning_rate.py:33`

*Released:* --lr_warmup is an absolute step count but T_0 is 30% of total iterations.

*Consequence:* Any run with 0.3*steps <= lr_warmup trips a message-less `assert warmup_steps < first_cycle_steps`. With the phase-2 default lr_warmup=500 that is any run under ~1667 optimiser steps.

*Fix:* Clamp to 10% of T_0 with a diagnostic naming both numbers.

*Evidence:* measured: condition computed

#### D07 · Correctness · `trainer.py`

*Released:* wandb imported and wandb.init() called unconditionally.

*Consequence:* The repo cannot run at all without a configured W&B account, even offline.

*Fix:* Optional import; --wandb_mode disabled. Flag: `--wandb_mode disabled`.

*Evidence:* measured: runs offline

---

### **High** — changes reported numbers or their interpretation (9)

#### D08 · Interpretability · `model.py (step-2 forward)`

*Released:* The three band-specific spatial-attention maps are computed but never enter the prediction.

*Consequence:* d(prediction)/d(spatial qkv) is exactly 0 for all three bands, so the maps interpreted as disorder signatures carry no label information. Their only training signal is the band-repulsion loss. The step-1 baseline DOES wire its spatial branch into the prediction (|grad| 406), so a step1-vs-step2 comparison is not an isolated test of frequency decomposition.

*Fix:* --spatial_head puts the maps on the prediction path. Flag: `--spatial_head`.

*Evidence:* measured: |grad| from prediction 0.0 vs from repulsion 2817

#### D09 · Interpretability · `visualization.py:103`

*Released:* The saved 'gradient' is d(-log S)/d(attention) -- the band-repulsion loss, not the prediction.

*Consequence:* For an L1-based S this is -[sign(h-l)+sign(h-u)]/(numel*S): five discrete values differing by one global scalar, 33% exactly 0. Magnitude carries no per-edge information and the diagnosis enters only through which subjects are averaged.

*Fix:* --attribution label_gradient (default) attributes the prediction; the published quantity stays available for reproduction. Flag: `--attribution`.

*Evidence:* measured: 4 distinct values over 2.56M entries, 33.4% zero

#### D12 · Evaluation protocol · `loss_writer.py:119 / metrics.py:24`

*Released:* The writer is constructed with Trainer.val_threshold (0) and nothing ever updates it, so Metrics.ROC_CURVE(name='test') thresholds sigmoid outputs at 0.

*Consequence:* Every subject is predicted positive: sensitivity 1.0, specificity 0.0, balanced accuracy 0.5, F1 = 2p/(1+p) -- all fixed by prevalence and independent of model quality. Reproduced on a model with AUROC 0.933.

*Fix:* The validation operating point is stored and carried over; a guard warns if the test threshold is outside (0,1).

*Evidence:* measured: released test 1.000/0.000/0.500/0.431 vs carried 0.891/0.866/0.878/0.794

#### D13 · Evaluation protocol · `dataloaders.py:367 / :3`

*Released:* Evaluation loaders use drop_last=True with a RandomSampler.

*Consequence:* (n mod batch) subjects are silently excluded from every evaluation, and because the sampler reshuffles, a different subset is dropped each epoch. At the repository's own ABIDE N=141 the released 70/15/15 split gives train 98 / val 21 / test 22, so the ABIDE fine-tuning script's --batch_size_phase2 16 discards 6 of 22 test subjects (27.3%) and 5 of 21 validation subjects (23.8%) at every evaluation -- a different 6 and 5 each epoch. At batch 32 the fold yields zero batches and the entire test set is skipped.

*Fix:* drop_last=False and SequentialSampler for eval folds.

*Evidence:* measured: fold sizes computed with the released train_test_split calls at N=141; AUROC SD 0.0025 across dropped subsets at n=300, B=16

#### D14 · Evaluation protocol · `pretrain_MBBN.slurm:38 + utils.py:129`

*Released:* The pretraining script passes --sequence_length_phase4 464 while running step 3; sort_args keeps only *_phase3 keys for step 3.

*Consequence:* The flag is silently discarded, pretraining runs at the phase-3 default 348, and loading that checkpoint into the 464-wide fine-tuning model raises RuntimeError on 4 of 44 shared keys. The published pretrain->finetune transfer therefore could not have run as scripted.

*Fix:* Corrected to --sequence_length_phase3; sort_args now warns when an explicitly-passed phase-tagged option is dropped.

*Evidence:* measured: RuntimeError reproduced through load_partial_state_dict

#### D17 · Training safety · `losses.py:37`

*Released:* -log(L1(h,l)+L1(h,u)+L1(l,u)) on row-stochastic maps, scaled by spatial_loss_factor up to 100.

*Consequence:* S is bounded above by 6/N and the bound is attained only at disjoint one-hot rows, so the loss's global optimum IS star/hub attention, independent of data. S=0 when the maps agree -- the state at initialisation -- so -log S is singular there and the gradient scales as 1/S (|grad|inf 0.50 at logit scale 1e-4 vs 1.6e-3 at scale 1).

*Fix:* --spat_diff_loss_type {minus_log (published), minus_log_eps, neg_linear, cosine} and --spat_diff_entropy_weight; --spatial_loss_warmup ramps lambda. Flag: `--spat_diff_loss_type`.

*Evidence:* measured: S ceiling 0.015 attained at one-hot; 400 steps move row entropy 0.9908->0.9625 of log N

#### D18 · Training safety · `trainer.py:632`

*Released:* NaN losses are replaced by 0.0 and training continues.

*Consequence:* A diverged run looks like a converged one; the loss curve simply flattens.

*Fix:* --nan_policy raise (default) / warn / zero (published). Flag: `--nan_policy`.

*Evidence:* code-read + test

#### D27 · Robustness · `model.py:188 / datasets.py:205`

*Released:* attention_mask=None is passed to the encoder, and ABIDE sequences are padded to a hardcoded 464 unconditionally with (pad//2, pad//2), which is one sample short for odd pad.

*Consequence:* The CLS pooling averages over pure-zero padding as if it were signal; a from-scratch 280-length ABIDE run is padded to 464 and fed to a model whose position embeddings stop at 281.

*Fix:* A real padding mask is threaded through; pad_to() is symmetric and returns a validity mask; padding is applied only when a pretrained checkpoint requires it (--pretrained_sequence_length). Flag: `--use_padding_mask`.

*Evidence:* code-read + test

#### D28 · Robustness · `model.py (step-3 masking)`

*Released:* The pretraining mask is deterministic and identical for every subject and every epoch: the top-k communicability ROIs plus every other fixed temporal window.

*Consequence:* At the released settings (--num_hub_ROIs 380 of 400, window 20, interval 2) almost the whole input is hidden, and the same positions every time, so the model can memorise the mask pattern rather than learn to infer missing signal.

*Fix:* --random_mask resamples per subject per epoch at --mask_ratio_spatial/--mask_ratio_temporal, fills with a learned mask token instead of 0, and --mask_loss_on_masked_only scores only hidden positions. Flag: `--random_mask`.

*Evidence:* code-read + test

---

### **Medium** — degrades training, robustness or cost (14)

#### D10 · Interpretability · `visualization.py:203`

*Released:* Hooks re-registered inside the per-subject loop and never removed; deprecated register_backward_hook; forward hook stores output[0] (subject 0) while the backward hook stores the whole batch.

*Consequence:* One hook pair accumulates per subject, and the saved activation/gradient pair have different shapes.

*Fix:* Registered once, removed in finally, register_full_backward_hook, consistent shapes.

*Evidence:* code-read

#### D11 · Interpretability · `visualization.py`

*Released:* No null distribution for a group difference in attention maps.

*Consequence:* An attention difference cannot be distinguished from chance.

*Fix:* --n_permutations label-permutation null with FWE correction over edges. Flag: `--n_permutations`.

*Evidence:* new capability

#### D15 · Evaluation protocol · `dataloaders.py (split)`

*Released:* train_test_split(stratify=y) only: no family grouping, no site stratification, and the split seed is the weight-init seed.

*Consequence:* ABCD contains siblings and twins, so relatives can straddle folds; ABIDE pools 17 sites with different scanners and TRs. Changing the init seed also repartitions the data, so seed-to-seed variance mixes both.

*Fix:* --group_by_family, --site_stratify, --leave_one_site_out, --split_seed. Flag: `--site_stratify`.

*Evidence:* new capability; degradation path measured

#### D16 · Evaluation protocol · `main.py / trainer.py`

*Released:* The test fold is scored every epoch alongside validation.

*Consequence:* Repeated test scoring invites selection on the test fold.

*Fix:* Selection is validation-only; the held-out fold is scored once at the end with the validation operating point. --eval_test_every_epoch restores the published behaviour. Flag: `--skip_final_test`.

*Evidence:* end-to-end test asserts single scoring

#### D19 · Training safety · `datasets.py`

*Released:* z-scoring divides by the raw std.

*Consequence:* A constant parcel (a dropout/coverage artefact) yields NaN for that ROI, which then reaches the nan_to_num above.

*Fix:* safe_zscore with an epsilon floor and an explicit zero-variance count.

*Evidence:* test pins NaN-free output

#### D20 · Training safety · `trainer.py:160`

*Released:* amp_state is referenced in save_checkpoint_ but never bound when AMP is off.

*Consequence:* NameError on the first checkpoint save of any non-AMP run.

*Fix:* Scaler state saved only when a scaler exists.

*Evidence:* code-read

#### D21 · Training safety · `trainer.py`

*Released:* --clip_max_norm is declared in main.py but never read by the trainer; clipping uses a hardcoded norm.

*Consequence:* The flag silently does nothing.

*Fix:* The declared value is used. Flag: `--clip_max_norm`.

*Evidence:* measured: 0 references in released trainer.py

#### D22 · Training safety · `trainer.py (accumulation)`

*Released:* optimizer.zero_grad() is called every step regardless of accumulation_steps.

*Consequence:* Gradient accumulation does not accumulate: the effective batch is unchanged and only the last micro-batch contributes.

*Fix:* zero_grad only at the accumulation boundary, loss scaled by 1/accumulation_steps. Flag: `--accumulation_steps`.

*Evidence:* code-read + test

#### D23 · Cost / performance · `datasets.py (__getitem__)`

*Released:* The per-subject knee-frequency fit (Lorentzian + Minuit spline) and three-band filtering run on every __getitem__.

*Consequence:* Identical work is repeated every epoch. Measured per subject: 273.8 ms at ABIDE's 360x280 (the Minuit fit converges slowly at 280 samples), 12.3 ms at ABCD 360x348, 16.1 ms at UKB 400x464.

*Fix:* --cache_bands memoises the decomposition to disk, keyed on dataset/subject/seq_len/TR/filter (427x / 16x / 18x per subject). Opt-in, and deliberately not the default: the per-subject ratio is large but the absolute saving is not, because the work is spread over 8 DataLoader workers and the cohorts are small in aggregate. Over the scripted epoch counts it saves 0.07 wall-hours on ABIDE (0.34 GB cache), 0.37 h on ABCD (27 GB) and 9.1 h on UKB pretraining (181 GB). It pays off for repeated runs over the same subjects -- sweeps, seed replicates, debugging -- not for a single training run. Flag: `--cache_bands`.

*Evidence:* measured

#### D24 · Cost / performance · `model.py:332`

*Released:* The communicability hub order is np.load-ed three times per forward pass, from a cwd-relative path, with the atlas name hardcoded.

*Consequence:* Disk I/O in the training loop, and the path breaks unless the job starts in the repo root.

*Fix:* Loaded once at construction, --communicability_dir/--communicability_dataset, generic ROI{n} fallback. Flag: `--communicability_dir`.

*Evidence:* code-read

#### D29 · Robustness · `model.py:56`

*Released:* Classifier = Linear -> BatchNorm1d(1) -> Dropout(0.6), applied to the scalar logit.

*Consequence:* One subject's logit moves SD 0.293 (range 1.98, sigmoid p spanning 0.107-0.464) from its batch-mates alone; a plain linear head gives SD exactly 0. Train-vs-eval mean logit gap 0.134, and Dropout(0.6) sets 61.4% of training logits to exactly 0.

*Fix:* --head_type linear (default) / published / mlp, shared by the baseline and MBBN so the architecture comparison is unconfounded. Flag: `--head_type`.

*Evidence:* measured

#### D30 · Robustness · `model.py (step-2 forward)`

*Released:* The three bands pass through the SAME temporal encoder with no band identity.

*Consequence:* The encoder is one identical function applied three times; nothing tells it which band it is reading.

*Fix:* --band_embedding adds a learned additive per-band embedding (3 x ROI parameters). Flag: `--band_embedding`.

*Evidence:* code-read + test

#### D31 · Robustness · `dataloaders.py (metadata joins)`

*Released:* Site and family attributes were resolved positionally out of a frame that is shorter than and reordered relative to the subject list; the ABIDE branch also re-pads ids to '0050001' while both metadata files carry 50001.

*Consequence:* Subjects would receive unrelated rows' attributes, and zip() truncates silently rather than raising. (Found while implementing D15, not in the released code, which had no site handling at all.)

*Fix:* canon_id() normalisation and id-keyed lookup with an explicit raise when no frame provides both columns.

*Evidence:* 3 regression tests

#### D32 · Robustness · `utils.py:100`

*Released:* weight_loader wraps its whole body in a bare `except:` and references undefined attributes in the step-1/3 branches.

*Consequence:* A missing or corrupt checkpoint, or any typo, silently becomes 'train from scratch'.

*Fix:* Explicit per-step handling; a nonexistent path raises FileNotFoundError.

*Evidence:* 2 regression tests

---

### **Low** — hygiene (5)

#### D25 · Cost / performance · `trainer.py:542`

*Released:* The checkpoint filename embeds the epoch number.

*Consequence:* Every improving epoch writes a NEW checkpoint (~325 MB for this model plus Adam state) instead of replacing the previous best.

*Fix:* Stable name; --keep_all_best_checkpoints restores the published behaviour. A last-epoch checkpoint is also written so a run whose validation never improves stays resumable. Flag: `--keep_all_best_checkpoints`.

*Evidence:* code-read

#### D26 · Cost / performance · `trainer.py:207`

*Released:* torch.compile is called in a loop over submodules but the result is discarded.

*Consequence:* The intended compilation never takes effect.

*Fix:* --compile_model actually installs the compiled module. Flag: `--compile_model`.

*Evidence:* code-read

#### D33 · Robustness · `utils.py:121`

*Released:* cudnn.deterministic=False and cudnn.benchmark=True are set inside reproducibility().

*Consequence:* The function named for reproducibility disables it.

*Fix:* --deterministic opts into deterministic cuDNN; the published throughput setting remains the default. Flag: `--deterministic`.

*Evidence:* code-read

#### D34 · Robustness · `datasets.py:66 / dataloaders.py:292 / loss_writer.py:48`

*Released:* Zero-variance ROIs tested with np.sum (which tests the mean); split files parsed with x[:-1] (truncates the last character of every line, not just the newline); `self.task == 'fine_tune' or 'bert' or 'test'` is always True.

*Consequence:* A zero-variance parcel passes the check; a split file without a trailing newline loses a character from the last subject id.

*Fix:* np.std, .rstrip('\n'), explicit membership test.

*Evidence:* code-read

#### D35 · Robustness · `dataloaders.py`

*Released:* Split files were written to a filename that differs from the one read back when new options are active.

*Consequence:* A regenerated split is not the split that gets loaded.

*Fix:* Single filename builder used for both write and read.

*Evidence:* end-to-end test asserts the written file is the one read

---

## Corrections made during this audit

Two claims were drafted, checked, and withdrawn. They are recorded because a reader
should be able to tell what was verified from what was merely plausible:

- **ABIDE split `KeyError`.** The released `elif self.target == 'ASD': self.target ==
  'DX_GROUP'` is a comparison whose result is discarded. This looked like a bug, but the
  ABIDE branch renames the `DX_GROUP` column to `ASD` *before* splitting and ABCD's
  metadata already uses `ASD`, so the no-op is correct and no error occurs. The statement
  is left in place as an explicit no-op; "fixing" it into an assignment would only change
  the split filename and invalidate existing split files.
- **Site/TR substring collision.** `_SITE_TR` is queried with `k in site`, which looked
  like it could match the wrong site. Enumerated: no key in `_SITE_TR` is a substring of
  any other, and the lookup returns the correct TR for every realistic ABIDE `SITE_ID`
  (`UM_1`, `USM`, `LEUVEN_2`, `MAX_MUN_a`, ...). Substring matching is in fact *required*,
  because SITE_IDs carry run suffixes. Only the silent `TR=3.0` fallback for an unlisted
  site was changed — it now warns, since TR sets the frequency axis and hence the bands.

One finding (D31) was introduced by this patch rather than found in the released code:
the first version of the site/family lookup resolved attributes positionally out of a
reordered frame. It is listed because the regression tests that pin it are part of the
deliverable.

## Reproducing the published behaviour

Defaults reproduce released behaviour except where that behaviour is itself the defect —
the test-set operating point (D12), the dropped evaluation subjects (D13), the mis-tagged
pretraining flag (D14), and the seven blocking bugs. Every other changed default is
reversible:

```bash
python main.py --step 2 \
    --spat_diff_loss_type minus_log \      # the published -log(S) objective, verbatim
    --head_type published \                # Linear -> BatchNorm1d(1) -> Dropout(0.6)
    --nan_policy zero \                    # substitute 0 for NaN and continue
    --eval_test_every_epoch \              # score the test fold every epoch
    --keep_all_best_checkpoints \          # one checkpoint file per improving epoch
    --no_padding_mask                      # attention_mask=None, as released
```

For the published attribution maps: `python visualization.py --attribution
spatial_difference`.

## Test suite

```bash
pip install pytest
python -m pytest tests -q
```

`tests/fixtures/synthetic.py` generates an ABIDE-shaped cohort (AR(1) parcel time series
with a weak class-dependent component, both metadata CSVs, and a communicability hub
order), so the suite — including an end-to-end `main.py` run — needs no cohort data, no
GPU and no Weights & Biases account. The released repository could not be executed at all
without all three.

## Cost measurements

Per-subject band decomposition (knee-frequency fit + three-band filtering), single
process, CPU:

| configuration | uncached | cached | speedup | cache/subject |
|---|---|---|---|---|
| ABIDE 360 280 | 273.8 ms | 0.64 ms | 427x | 2.42 MB |
| ABCD 360 348 | 12.3 ms | 0.76 ms | 16x | 3.01 MB |
| UKB 400 464 | 16.1 ms | 0.89 ms | 18x | 4.46 MB |

The ABIDE configuration is ~20x more expensive per subject than the longer series because
the Minuit `f2` fit converges slowly at 280 samples.

Projected over the cohort sizes in this repository's own README (UKB 40,699 + ABCD 8,833 +
ABIDE 141 = 49,673, matching the total stated in the paper abstract) and the epoch counts in
the launch scripts, dividing the serial cost by the 8-worker `DataLoader` cap:

| cohort | N | epochs | cache saves | cache size |
|---|---|---|---|---|
| ABIDE (`finetune_MBBN.slurm`) | 141 | 50 | 0.07 wall-h | 0.34 GB |
| ABCD (`train_MBBN_from_scratch.slurm`) | 8,833 | 100 | 0.37 wall-h | 26.57 GB |
| UKB (`pretrain_MBBN.slurm`) | 40,699 | 400 | 9.09 wall-h | 181.34 GB |

**Honest verdict: the per-subject ratio is large but the absolute saving is not.** The work
is spread over 8 workers and the cohorts are small in aggregate, so even UKB pretraining —
400 epochs over 40,699 subjects — recovers about 9 wall-hours for a 181 GB cache. This is
why `--cache_bands` is opt-in. It earns its keep for *repeated* passes over the same
subjects (hyperparameter sweeps, seed replicates, debugging), not for a single training run.

## Cohort-dependent consequences of the evaluation defects

Using the same N=141 for ABIDE, the released `train_test_split` calls at 70/15/15 give
train 98 / val 21 / test 22. With `drop_last=True` on the evaluation loaders (D13):

| batch size | val subjects dropped | test subjects dropped | test batches |
|---|---|---|---|
| 4 | 1 of 21 | 2 of 22 | 5 |
| 8 | 5 of 21 | 6 of 22 | 2 |
| 16 | 5 of 21 | 6 of 22 | 1 |
| 32 | 21 of 21 | 22 of 22 (fold skipped entirely) | 0 |

`finetune_MBBN.slurm` uses `--batch_size_phase2 16`, so 6 of 22 test subjects (27.3%) and 5
of 21 validation subjects (23.8%) were excluded from every evaluation — and because the
evaluation loader used a `RandomSampler`, a *different* 6 and 5 each epoch. At batch 32 the
test fold yields zero batches.

## GPU validation

The mixed-precision and CUDA code paths cannot be reached on a CPU-only machine, so
the suite and three short training runs were also executed on an NVIDIA GB10
(compute capability 12.1, torch 2.14.0a0+4fdf77b940.nv26.08,
fp32 matmul precision `tf32`) inside the NGC PyTorch container, on
the same synthetic cohort.

| run | result |
|---|---|
| regression suite | 58 passed, 1 warning in 82.36s (0:01:22) |
| step 2, mixed precision on, grad clipping + accumulation | exit 0, 80 s |
| step 2, mixed precision off | exit 0, 57 s |
| step 3, random masking + masked-only reconstruction loss | exit 0, 44 s |

This pass also caught a defect introduced by the patch itself: the guarded NVTX helper
called itself instead of `torch.cuda.nvtx`, which is invisible on CPU because the
`is_available()` guard short-circuits, and fatal with a CUDA device present
(`RecursionError`). `tests/test_protocol.py` now pins both the delegation and the
absence of self-recursion, and asserts statically that every CUDA-only call in
`trainer.py` sits inside a function that tests `torch.cuda.is_available()`.
