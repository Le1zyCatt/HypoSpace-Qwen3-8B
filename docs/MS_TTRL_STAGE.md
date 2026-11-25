# Preliminary MS-TTRL adaptation

English synthesis of the teammate's stage report, dated25 November2026,16:20 China Standard Time. These are teammate-reported results; the original raw archives have not been supplied for independent re-audit in this repository. Incomplete results are not treated as completed experiments.

## Setup and result

Qwen3-8B;24 training tasks, eight/domain;26 frozen evaluation tasks(Boolean10, Causal8,3D8). Both trained arms start from the same base model. Completed:seed1337, epoch10, Boolean/Causal only. Planned:30 epochs Boolean,20 Causal/3D; seeds1338/1339 pending.

Majority TTRL samples eight candidates and rewards the most frequent parseable canonical answer. MS-TTRL samples eight candidates and eight same-policy verifier votes per unresolved outcome. Fixed denominator8:7–8 positive votes confirm,5–6 remain pending,0–4 reject. State/frequency-based rewards feed GRPO. Evaluation usesN4 and matched decoding settings/seeds; cross-domain aggregation weights tasks equally. The report does not fully specify thinking, LR, response cap, precision, or optimizer details.

|Domain|Method|VR %|NR %|RR@4 %|
|---|---|---:|---:|---:|
|Boolean10|Base|85.00|32.50|25.44|
||Majority|85.00|35.00|24.33|
||MS|92.50|30.00|22.94|
|Causal8|Base|100.00|43.75|12.79|
||Majority|96.88|46.88|15.52|
||MS|96.88|40.63|15.56|
|Combined18|Base|91.67|37.50|19.82|
||Majority|90.28|40.28|20.41|
||MS|94.44|34.72|19.66|

Combined MS−Base:ΔVR+2.78pp,ΔNR−2.78pp,ΔRR−0.16pp. MS−Majority:+4.17,−5.56,−0.76pp. Deltas use unrounded values. Boolean:37/40 valid versus34/40 for controls; four Base truncations become valid while other positions lose one valid answer. Causal:both trained arms31/32 valid versus Base32/32; one task recovers two extra outcomes(+50pp within-taskRR), while the other seven have three ties and four declines.

## Reported cost and verifier audits

First10 epochs, excluding evaluation:MS/majority Boolean46.2/8.3min, Causal79.6/7.9min, approximately5.6×/10.1×. Extra verifier votes:320+432=752. All64 first-epoch candidates/domain matched between arms. The teammate reports auditing all votes and recomputing majority rewards for1,280 candidates.

Boolean produced40 distinct task/outcome pairs:20 confirmed answers satisfied observations;20 rejected answers violated them. Four confirmed expressions were missing from the reference set:one had a functionally equivalent reference; three had none. A binary-GT versus multi-child-parser depth/grammar mismatch was identified; its frozen-test impact is not quantified. Causal produced53 pairs:six confirmed and47 rejected agreed with structural/intervention recomputation and trainingGT. TrainingGT was used for diagnostics; frozen-testGT only for scoring.

## 3D diagnostic pilot

On the same41 observed training candidates, old/revised verifiers each generated328 votes. Coordinate representation, verdict-conflict checking, and parsing changed together.

|Candidate type|Count|Old confirmed/pending/rejected|Revised confirmed/pending/rejected|
|---|---:|---|---|
|Valid|26|4/12/10|24/2/0|
|Invalid|15|0/1/14|0/0/15|

Unparsed votes fell39/328→19/328. Revised epoch1:about68.1min, including candidate generation12.9, verification54.3, update0.8min. All eight training groups had reward differences and nonzero gradients. Because prompts/parsing were revised on these known outputs, this measures joint diagnostic improvement, not held-out generalization. Formal3D evaluation remains pending.

## Interpretation

The generation→verification→cache→reward→update chain runs. This stage improves validity, not demonstrated coverage. Complete3D, final checkpoints, and extra seeds remain necessary.

The [paper](https://arxiv.org/html/2610.32234v1) publicly providesG8, LR1e-6, batch16 prompts, response cap1024,20/30 epochs, and training splits300 causal/1003D/50Boolean; evaluationk differs fromN4. The [current README](https://github.com/zzx-peter/Multiple-Solution-TTRL) excludes unparseable votes from the denominator, unlike this fixed-eight pilot. Before blaming undisclosed settings, align upstream commit, effective script arguments, denominator, GT grammar, budgets, and evaluationN. Current evidence does not isolate a causal explanation for the performance gap.
