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

Evaluation uses the same shared repeated-evaluation runner as
`src/evaluation/evaluate_pusht.py`: three repeats of 50 episodes with base seed
42 by default. The resulting non-overlapping seed ranges are 42--91, 92--141,
and 142--191. Use `evaluate --force` (or `all --force-evaluate`) to replace
results produced by the former single-run 200-episode protocol; caches and
trained heads are reused.

The ablation keeps its predefined 96x96 policy observations and records that
choice as `config.observation_resolution` in every evaluation. Override it
explicitly with `--observation-resolution`; production evaluation defaults to
224x224. Results at different observation resolutions are separate protocols
and should not be pooled.

The spatial-detail extension adds four predefined zero-shot conditions:

- `resolution_224`: render the same clean state directly at 224x224.
- `blur_1`, `blur_2`, and `blur_4`: deterministic Gaussian blur with the named
  pixel-space sigma, applied after the baseline 96x96 rendering.

The default `all`, `evaluate`, and `analyze` commands select only LeWM. They
cache paired latents and evaluate its clean-trained heads on all four
conditions. The secondary adaptation suite additionally trains LeWM heads on
`blur_4` and evaluates them on clean and matched blurred inputs. This keeps the
resolution comparison zero-shot and prevents it from being conflated with
adaptation. DINOv2 remains opt-in with `--encoders lewm dinov2`; analysis filters
unselected encoder artifacts, including older results already on disk.

`analyze` computes latent shift directly from the cached, counterfactually
paired expert states. Results are written to `analysis/latent_metrics.csv` and
to the `latent_metrics` table in `analysis/report.json`, including mean cosine
similarity, fifth-percentile cosine, normalized L2 change, and variance ratio.
It also writes clean-head action MAE/RMSE to `analysis/action_metrics.csv`.
Analysis rejects mixed episode/repeat/seed protocols and mixed resolutions for
the same named condition, so stale 200-episode results cannot be silently
pooled with the new 3x50 suite.

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
