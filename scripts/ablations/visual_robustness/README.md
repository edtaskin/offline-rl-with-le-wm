# LeWM PushT visual-robustness ablation

This package is deliberately separate from the production BC and evaluation
CLIs. It reuses their stable policy, history, environment, and evaluator
interfaces while owning all visual interventions and artifacts.

Run the predefined experiment:

```bash
python -m scripts.ablations.visual_robustness all
```

Run stages independently:

```bash
python -m scripts.ablations.visual_robustness cache
python -m scripts.ablations.visual_robustness train --encoders lewm dinov2 --conditions clean
python -m scripts.ablations.visual_robustness evaluate --train-conditions clean
python -m scripts.ablations.visual_robustness analyze
```

Generate only the paired evaluation-environment screenshots:

```bash
python -m scripts.ablations.visual_robustness screenshots
```

This writes two reset frames per condition plus a labeled contact sheet under
`environment_screenshots/`. All conditions use the same reset seeds and the
same fixed-target, radius-200 evaluation setup.

The DINOv2 adapter is offline-only: it expects the local torch-hub repository
and `dinov2_vits14_pretrain.pth` checkpoint. Override those locations with
`--dinov2-repo` and `--dinov2-checkpoint`.

All generated artifacts are written below
`runs/ablations/lewm_visual_robustness/` by default.

