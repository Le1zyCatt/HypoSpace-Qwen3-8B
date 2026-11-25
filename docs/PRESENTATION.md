# Four-speaker ten-minute presentation

12 slides, 600 seconds; A/B/C/D each 150 seconds. Replace letters with names. Rehearse at about 140 spoken seconds per speaker, leaving ten seconds for pauses and transitions.

|Speaker|Slides|Time|Topic|
|---|---|---|---|
|A|1–3|0:00–2:30|Question,metrics,inference|
|B|4–6|2:30–5:00|GAPO,MS-TTRL,verification|
|C|7–9|5:00–7:30|Distillation and E16|
|D|10–12|7:30–10:00|Follow-up,analysis,conclusions|

## 1. Qwen3-8B on HypoSpace

Speaker A; 35 seconds; 0:00–0:35

### On-slide content

- Valid-outcome coverage and reasoning efficiency
- GAPO, process distillation, and preliminary MS-TTRL

### Speaker notes

Our task has multiple correct answers. Producing one correct answer repeatedly is therefore insufficient: the model must recover different valid mechanisms. We investigated inference settings, group-aware rewards, reasoning distillation, and self-verification. Our main case study is the first held-out 36-task confirmation. Outputs became substantially shorter, while coverage improved numerically but not conclusively. We also report the attempts that did not help and the later tests that limit our conclusions.

Sources/evidence: HypoSpace: https://arxiv.org/abs/2510.15614 ; E16 results/e16/summary.json

## 2. Validity, repetition, and coverage

Speaker A; 60 seconds; 0:35–1:35

### On-slide content

- Finite observations can admit several valid mechanisms
- Complete GT → parsing → canonicalization → scoring
- GT={A, B, C}; samples=A, A, A, A: VR=100%, RR=1/3

|Metric|Definition in our later IID protocol|
|---|---|
|VR|Valid candidates / N|
|NR|Distinct parseable outcomes / N; errors may count|
|RR@N|Distinct valid mechanisms / complete GT size|

### Speaker notes

HypoSpace constructs observations and exhaustively enumerates compatible mechanisms. We use causal graphs, 3D structures, and Boolean expressions. Suppose the valid set contains A, B, and C. Four copies of A achieve perfect validity but recover only one-third of the set. NR measures distinct parseable outcomes, and can increase because of different wrong answers; RR is therefore essential. We separate task families before building multiple training answers, and compare models on the same tasks and budgets. Mechanism identity also matters: logical equivalence is not always the benchmark’s definition of the same Boolean mechanism.

Sources/evidence: HypoSpace repository; strict GT scorer src/reward.py

## 3. Thinking helps; default DSH does not reliably help

Speaker A; 55 seconds; 1:35–2:30

### On-slide content

- Eight conditions: 8B/27B × thinking × direct/DSH
- Early small tasks and N=GT; not comparable to later N=4
- Successful DSH candidates usually took one model step

|Early 8B RR %|Causal|3D|Boolean|
|---|---|---|---|
|No thinking, direct|22.08|3.40|0.00|
|Thinking, direct|87.50|62.96|46.14|
|Thinking, DSH|79.86|54.32|38.99|

### Speaker notes

We first worried that an 8B model would score too poorly to distinguish methods. The eight-condition comparison showed meaningful headroom once thinking was enabled. The default DSH wrapper did not consistently improve recovery, and successful candidates usually involved only one model step. That constrains this wrapper configuration, not all iterative search methods. These early tests used smaller tasks, history-conditioned generation, andN equal to ground-truth multiplicity. Later absolute scores must not be compared directly. This baseline motivated continuing with 8B rather than simply switching to a larger model. SpeakerB will now explain the reward-based attempts.

Sources/evidence: Early inference reportE02; @deepseek-ai/dsh0.2.2-rc.1 engineering baseline

## 4. GAPO works on lists; transfer remains limited

Speaker B; 50 seconds; 2:30–3:20

### On-slide content

- Reward low-frequency valid outcomes within a group
- List: BF16/G32/200 updates; Hypo: NF4/G8/40 updates
- Hypothesis: reward reweighting cannot create unseen support

|Experiment|Base|GAPO|Finding|
|---|---|---|---|
|List: distinct valid answers|1.354|2.594–2.813|All three seeds improve|
|HypoSpace: RR %|18.77|18.74|No gain over Base|

### Speaker notes

GAPO makes rewards depend on the frequency of valid outcomes within a group. In our teammate’s list-task replication, all three training seeds increased the number of distinct valid answers. This is positive evidence that the mechanism can reshape an output distribution. Direct transfer to our harder HypoSpace setup did not beat Base: RR was 18.74 versus 18.77. There were only three independent test tasks per domain, so this is a narrow result. One possible limitation is support: rewards can reweight answers that appear, but cannot automatically teach reasoning for valid mechanisms never sampled.

Sources/evidence: Anschel et al.,EMNLP2025 https://aclanthology.org/2025.emnlp-main.1649/ ; E05/E07

## 5. MS-TTRL: higher validity, no coverage advantage yet

Speaker B; 45 seconds; 3:20–4:05

### On-slide content

- Same-policy verification + cache + rare-outcome rewards
- One seed, two domains; formal 3D evaluation pending
- Training cost: 5.6–10.1× the majority baseline

|18 tasks, epoch 10|VR %|NR %|RR@4 %|
|---|---|---|---|
|Base|91.67|37.50|19.82|
|Majority TTRL|90.28|40.28|20.41|
|MS-TTRL|94.44|34.72|19.66|

### Speaker notes

MS-TTRL separates correctness estimation from generation frequency. The policy verifies candidate outcomes, while frequency promotes rare reliable solutions. Our teammate has a preliminary comparison at epoch 10, seed 1337, on 18Boolean and causal tasks. MS-TTRL has the highest validity, but its recovery does not beat Base, and majority TTRL has the best coverage here. Verification also makes training roughly five to ten times slower. Since 3D evaluation and additional seeds remain incomplete, we report a working training chain and partial evidence, not reproduction of the paper’s large headline gains.

Sources/evidence: Zhang et al.,MS-TTRL https://arxiv.org/abs/2610.32234 ; docs/MS_TTRL_STAGE.md

## 6. Verifier semantics and protocol alignment

Speaker B; 55 seconds; 4:05–5:00

### On-slide content

- Observation consistency can differ from finite-GT membership
- 3D revision: valid structures confirmed 4→24 out of 26
- Known gaps: training scale, epoch, N=4, and vote denominator
- Undisclosed details are a question, not an established cause

### Speaker notes

Verification introduces a second boundary. Some Boolean expressions satisfy observations but fall outside the finite reference grammar. In 3D, revising representation and parsing increased confirmed valid structures from 4 to 24 out of 26, on the same observed training candidates. That is a diagnostic repair, not independent generalization. The paper publicly provides major settings; our teammate’s training scale, epoch 10 checkpoint, N 4 evaluation, and fixed voting denominator already differ from the paper or current code. We should align these before attributing the reproduction gap to missing details. SpeakerC will now explain why we moved from rewards to process distillation.

Sources/evidence: MS-TTRL AppendixB/C and official README; teammate stage report

## 7. Short explanations exposed a validity trade-off

Speaker C; 45 seconds; 5:00–5:45

### On-slide content

- 96 families/288 QA; 72 updates per SFT arm
- Teacher content targets, not full reasoning traces
- Hypothesis: shorter targets suppress useful checks

|18 dev tasks, N=8|VR %|RR %|Mean tokens|
|---|---|---|---|
|Base|76.39|14.77|5,233|
|Answer-only SFT|59.03|14.21|63|
|Short-teacher SFT|36.11|14.55|206|

### Speaker notes

We first compared answer-only supervision with short teacher explanations. Using 96 families and 288 QA pairs, we trained each arm for 72 updates. Outputs shrank from over five thousand tokens to tens or hundreds, but validity fell sharply and recovery did not improve. Shortness was clearly insufficient. The targets were teacher content, not full reasoning traces. Our trace audit also found repeated confirmations, incorrect deductions, and a bias toward simple answers. This motivated editing the student’s own reasoning rather than simply teaching a jump to the final answer.

Sources/evidence: DeepSeek-R1 distillation motivation https://arxiv.org/abs/2501.12948 ; E08/E09; short-target design is custom

## 8. DRP-inspired multi-outcome process distillation

Speaker C; 50 seconds; 5:45–6:35

### On-slide content

- Student traces → teacher editing → mechanism checks → SFT
- Initial DRP attempt failed validity/coverage gates
- Nine-arm diagnosis separates process type and ending weights
- Selected: 48 families, 144 QA, 72 updates, LR 2e-5, end weight 3

### Speaker notes

DRP suggests editing student reasoning before distillation. We adapted that idea with mechanism checks and targeted multi-outcome augmentation. The first attempt shortened outputs but harmed both validity and coverage. We then examined NF4 versus BF16 deployment, and compared four process types with two ending-token weights, plus an answer-only control. A 12-task development screen selected diverse_w. The selected model uses 48 families, 144 QA pairs, 72 updates, and a threefold weight on end-of-thinking and EOS tokens. This combines editing, repair, and augmentation; we cannot attribute the eventual result exclusively to pruning sentences.

Sources/evidence: Jiang et al.,DRP https://aclanthology.org/2026.findings-acl.196/ ; E10–E15

## 9. Primary result: 27.5% fewer tokens; positive RR signal

Speaker C; 55 seconds; 6:35–7:30

### On-slide content

- RR: +2.71 pp; 95% CI[−0.59,+5.96]; Holm p=0.258
- Tokens: −1,457; 95% CI[−1,671,−1,245]
- Length reduction is supported; RR improvement is unconfirmed

|36 held-out tasks,8,192 cap|Base|Process distillation|
|---|---|---|
|VR %|75.00|76.39|
|RR@4 %|15.57|18.28|
|Mean output tokens|5,303|3,846|
|Truncated %|6.94|1.39|

### Speaker notes

This is our primary result. The checkpoint was selected before freezing 36 unseen task families, with four samples per task. At the 8, 192-token cap, recovery rises from 15.57 to 18.28, and mean output falls from 5, 303 to 3, 846, a 27.5 percent reduction. The token-difference interval lies clearly below zero. The RR interval crosses zero, and its adjusted p-value is 0.258, so we call it a positive coverage signal, not a significant improvement. In 3D, validity improves while coverage is unchanged, again separating validity from diversity. SpeakerD will explain the longer-budget and larger-scale follow-up.

Sources/evidence: E16 results/e16/summary.json; token intervals are unadjusted secondary analyses

## 10. More budget and data did not ensure more coverage

Speaker D; 50 seconds; 7:30–8:20

### On-slide content

- E16 at 40K: RR remains unconfirmed; no extra valid method samples
- Expansion does not beat the old 48-family checkpoint
- High LR can create wrong short answers or 40K reasoning tails

|Matched comparison|Base RR %|Method RR %|Token change|
|---|---|---|---|
|E16, 40,960 cap|17.11|18.28|−23.9%|
|96 tasks, 8K, 192-family mean|12.90|14.37|−19.1%|
|96 tasks, 40K, 192-family mean|13.01|14.41|−0.7%|

### Speaker notes

At 40, 960 tokens, E16 still saves 23.9 percent, but its RR advantage becomes smaller and remains unconfirmed. We also scanned learning rate and answer weighting, and expanded to 192 families, three training seeds, and 96 new held-out tasks. The expanded mean is slightly above Base, but not significant after correction, and does not exceed the earlier 48-family model. At 40K, its token savings almost disappear. Aggressive learning rates produced either empty reasoning with wrong answers or very long incorrect traces. More updates, data, and tokens are not automatic gains. Rows here are within-experiment comparisons, not a cross-dataset ranking.

Sources/evidence: E16; results/secondary/final96_summary.json; aggressive-LR failure auditE23

## 11. Valid mass, outcome concentration, and sample cost

Speaker D; 55 seconds; 8:20–9:15

### On-slide content

- IID model: E[RR@N]=(1/L) Σₖ[1−(1−pₖ)ᴺ]
- Validity: mass inside GT; coverage: mass across different mechanisms
- Rewards reshape support; multi-outcome targets may expand it
- Shortening one answer’s trace does not create new mechanisms

### Speaker notes

A simple IID model helps explain the results. If a mechanism has generation probability p, its chance of appearing at least once inN samples is one minus one minus p raised toN. Coverage depends both on valid probability mass and its concentration across outcomes. Frequency rewards reweight observed support; multi-outcome targets may expand it; process editing reduces sample cost. These are different operations. Self-verification can also amplify wrong cached labels. The formula explains later independent sampling, not the original history-conditioned loop. The next useful experiment should match total token or time cost, rather than assume shorter output automatically improves recovery.

Sources/evidence: Our probabilistic analysis under IID; MS-TTRL frequency-conflict motivation

## 12. Conclusions and reproducible outputs

Speaker D; 45 seconds; 9:15–10:00

### On-slide content

- Positive evidence: list-task reward mechanism; E16 output reduction
- Boundaries: transfer, short targets, high LR, and scale remain constrained
- Release: frozen tasks, raw token traces, scores, and CPU verification
- Next: equal-cost sampling and orthogonal process/target controls

### Speaker notes

Our contribution is an auditable chain of experiments. Group-aware rewards work on lists; E16 clearly compresses outputs and gives a positive but unconfirmed coverage signal. Direct transfer, short explanations, and expansion do not establish stable coverage gains. The repository includes frozen tasks, raw token trajectories, task-level scores, and CPU verification. Next, we should spend saved tokens on extra sampling and independently vary process editing and multi-outcome targets. The unsuccessful attempts are useful because they identify which assumptions do not hold under our tested configurations.

Sources/evidence: README.md; docs/REPORT.md; docs/REPRODUCE.md

## Questions to prepare

**Why E16?** A clearly interpretable held-out efficiency case. The later96-task result is disclosed and limits stable-coverage claims.

**Does failure disprove a method?** No;these are configuration-specific bounds. Truncation-limited zero scores and incomplete runs are not evidence of universal failure.

**Is this an exact reproduction?** GAPO/DRP are cross-model/task adaptations. MS-TTRL is a partial implementation with known protocol gaps.

**Did pruning cause the gain?** Not uniquely established;process editing,target diversity,and ending supervision change together.

**What is significant?** Output length reduction has intervals below zero;RR improvement is not statistically confirmed.
