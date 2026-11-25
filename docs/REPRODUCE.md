# Reproducing E16

## CPU statistical verification

Follow the README and run `python src/analyze.py`. The script checks 144 score records, cell means, task-paired intervals, and two-budget Holm-adjusted RR tests. Bootstrap: seed 20261121, 20,000 domain-stratified task resamples. Sign-flip test: seed 20261122, 100,000 draws. A task, not an individual candidate, is the independent unit. With 12 tasks/domain, task and domain macro means coincide.

VR=valid samples/N; NR=distinct parseable outcomes/N, including invalid outcomes; RR=distinct strict-GT mechanisms recovered/complete GT size. Boolean identity uses the upstream mechanism canonicalizer rather than arbitrary functional equivalence.

To regenerate the figure,install `requirements-figures.txt` and run `python src/plot.py`.

## Re-parse archived answers

```bash
git clone https://github.com/CTT-Pavilion/_HypoSpace third_party/_HypoSpace
git -C third_party/_HypoSpace checkout c69e9318577b34b5b896996571aefd4ba6053f58
python src/rescore.py
```

Only `boolean/boolean_dataset.py` is dynamically imported from upstream; it requires SymPy. Add `--tokenizer /path/to/Qwen3-8B` to decode all 576 scored prefixes from the 288 archived long token trajectories and check seeds, lengths, and final strings. This needs the tokenizer but no model inference. Without that option, canonical parsing, GT membership, and deduplication are verified.

## Retrain the selected configuration

Historical environment: Python 3.12, CUDA-enabled PyTorch 2.11.0; remaining versions are pinned in `requirements-training.txt`. Install a PyTorch build compatible with the GPU/CUDA stack, then verify versions. These are portable reproduction commands; publication did not trigger another GPU training run.

```bash
pip install -r requirements-training.txt
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir /path/to/Qwen3-8B
python src/train.py --model /path/to/Qwen3-8B --output outputs/e16-retrain
python src/merge.py --model /path/to/Qwen3-8B --adapter outputs/e16-retrain/adapter --output outputs/e16-merged
```

Actual settings: NF4 double quantization, BF16 compute; LoRA r32/alpha32/dropout0 on q/k/v/o/gate/up/down projections; 87,293,952 trainable parameters; micro-batch1, accumulation8; LR2e-5, cosine, warmup4, weight decay0, max gradient norm1. Four logical epochs are expanded to576 ordered sequences and trained for one physical epoch,72 updates. Max sequence5,343; no truncation. Prompt labels are masked. End-of-thinking token151668 and EOS151645 receive weight3, other supervised tokens weight1; loss is normalized within each example, then averaged. Here `w` refers to ending-token weighting, not answer-content weighting.

The exact ordered tokens/labels are included, avoiding teacher-API regeneration cost. `training_core.py` preserves the historical objective/model construction; `train.py` replaces machine-specific orchestration and profiling. Historical training took665 seconds on one5090, excluding data production and evaluation. Adapter/optimizer weights remain in private archives; retraining is supported but bitwise identity is not promised.

## Generate new paired evaluations

```bash
python src/evaluate.py --model /path/to/Qwen3-8B --arm base --output outputs/new-eval/base
python src/evaluate.py --model outputs/e16-merged --arm best --output outputs/new-eval/best
```

Use `--limit 1` for a one-task connectivity check. These commands do not run automatically and may take hours. BF16 merged, batch1, thinking on, temperature0.6, top-p0.95, top-k20, min-p0; N4 IID, no candidate history or tools. Seeds are `(20261113 + int(SHA256('confirm:qid:index')[:8],16)) mod 2^32`. Score both caps from one40,960-token trajectory per model/task/sample; do not treat them as independent calls.

Default original RoPE is unchanged; runtime capacity is41,745. The historical evaluation reports overruns of native32,768 and original configured40,960 context limits; the best model has one of144 long trajectories beyond those thresholds. This is not a claim of new long-context training. Historical evaluation distributed work over three5090s and six4090s and counterbalanced model order by worker parity. The portable entry uses one GPU and does not reproduce orchestration, heartbeats, or timing.

New outputs are written separately and must not overwrite archived E16 scores. Aggregate complete matched36-task/two-model/two-budget outputs using the same scoring/analysis, not partial connectivity samples.

## Splits and limits

Train48 families, selection-dev12, held-out test36. Causal5 variables; 3D5×5/H3; Boolean3 variables/depth2. E16 froze test inputs after checkpoint selection; see config/final_audit. After inspection, those36 tasks cannot serve as a new sealed test, hence the later96-task set.

This release supports training from frozen targets, new generation, and verification of existing results. The complete teacher-editing/manual-review process is not packaged as one automatic pipeline. Frozen records are reused rather than redrawing easier tasks. No same-QA unpruned full-CoT control or same-size192-family answer-only control was completed, so pruning's isolated causal contribution is unresolved.
