# Sources and license boundaries

This is a Qwen3-8B adaptation and empirical study, not a reproduction of every original model, dataset, or headline score.

- HypoSpace: https://github.com/CTT-Pavilion/_HypoSpace ; commit `c69e9318577b34b5b896996571aefd4ba6053f58`. Upstream source is not vendored. Check out the original repository to preserve attribution and license; its Boolean canonicalizer is used for re-scoring.
- Base model: https://huggingface.co/Qwen/Qwen3-8B ; revision `b968826d9c46dd6066d109eabc6255188de91218`. Weights remain subject to the upstream license.
- GAPO: https://aclanthology.org/2025.emnlp-main.1649/
- DRP: https://aclanthology.org/2026.findings-acl.196/
- MS-TTRL: https://arxiv.org/abs/2610.32234 ; code https://github.com/zzx-peter/Multiple-Solution-TTRL
- DSH wrapper: `@deepseek-ai/dsh 0.2.2-rc.1`, an engineering baseline, not a method proposed here.
- Reasoning-distillation motivation: https://arxiv.org/abs/2501.12948
- Process-supervision motivation: https://arxiv.org/abs/2305.20050 . We did not train a PRM.

Generated tasks and teacher-/human-edited process targets are provided for academic inspection. No third-party PDFs or model weights are bundled. A new open-source license has not been selected on the team's behalf; upstream components retain their respective terms.
