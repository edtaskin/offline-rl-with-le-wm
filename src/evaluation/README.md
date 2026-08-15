# PushT evaluation protocol

[`evaluate_pusht.py`](evaluate_pusht.py) is the common real-environment evaluator for latent BC,
latent PPO, state-BC, and CNN-BC checkpoints. It writes the complete result to `metrics.json` and
prints the scalar and per-stratum summaries to the terminal.

## Canonical protocols

`canonical_v1` preserves the historical, distribution-matched seed suite. `canonical_v2` is the
difficulty-balanced protocol intended for checkpoint comparison. It constructs one deterministic
suite of simulator reset seeds, rejects starts that already satisfy the task, and distributes the
episodes as evenly as possible over six initial-pose strata. At the standard 150 episodes, every
stratum contains exactly 25 starts.

A **stratum** is one initial-state bucket in the Cartesian product of block-to-goal translation and
wrapped absolute rotation error:

| Centroid translation (`block_goal_dist`) | Rotation (`block_angle_dist`) | Stratum |
| --- | --- | --- |
| less than 70 px | less than 45 degrees | `near_aligned` |
| less than 70 px | at least 45 degrees | `near_misaligned` |
| 70 to less than 140 px | less than 45 degrees | `mid_aligned` |
| 70 to less than 140 px | at least 45 degrees | `mid_misaligned` |
| at least 140 px | less than 45 degrees | `far_aligned` |
| at least 140 px | at least 45 degrees | `far_misaligned` |

The stratum describes only the starting block pose. It does not describe the number of steps used
or how the episode eventually terminates. Translation is measured between the geometric centers of
the block and goal T. This matches the radius used by the start sampler and avoids counting some of
the T body's off-center pose-origin shift as both translation and rotation.

## Success metrics

Let `success_rate_s` be the fraction of successful episodes in stratum `s`.

- `success_rate` is the successful fraction over all episodes.
- `balanced_success_rate` is the unweighted mean of the six `success_rate_s` values. It prevents a
  more numerous or easier stratum from dominating the score. With exactly 25 episodes in every
  stratum, it equals the ordinary `success_rate`.
- `hard_success_rate` is `success_rate_far_misaligned`, where `far_misaligned` is the protocol's
  designated hardest starting stratum. It is not defined as the minimum observed score.
- `worst_stratum_success_rate` is the minimum of the six observed stratum success rates. It can
  differ from `hard_success_rate`.
- `success_by_N` is the fraction of all episodes completed successfully within `N` environment
  steps. It is cumulative, so `success_by_50 <= success_by_100 <= success_by_200 <=
  success_by_300`. At the 300-step horizon, `success_by_300` equals `success_rate`.
- `balanced_success_by_N` is the unweighted mean of `success_by_N` across the six strata.

For example, `success_by_50 = 0.30` means 30% of all episodes were solved in at most 50 steps. It
does not mean that step 50 or the corresponding starts were easier than step 300; 50 is simply the
stricter completion budget.

## Success AUC

`success_auc` is the normalized area under the cumulative success-by-step curve over horizon `H`.
For an episode successfully completed at length `L`, its contribution is

```text
(H - L) / H
```

A failed episode contributes zero. Thus early successes receive more credit, while a failure or a
success exactly at the horizon contributes zero. The final score is the mean contribution over all
episodes and lies between zero and one. It combines reliability and speed; it is not itself a
success probability.

`balanced_success_auc` first computes this AUC separately in every stratum and then takes the
unweighted mean. With equal stratum sizes it equals `success_auc`.

## Checkpoint selection

A useful lexicographic selection order is:

1. Maximize `balanced_success_rate` for overall difficulty-balanced reliability.
2. Maximize `hard_success_rate` or `worst_stratum_success_rate` to reject checkpoints with a weak
   difficult-start regime.
3. Use `balanced_success_auc` as an efficiency tie-breaker between similarly reliable checkpoints.

This keeps a fast but less reliable policy from outranking a slower policy that solves materially
more starts.

## Visualizing the deterministic starts

Pass `--visualize-starts` to write `start_locations.png` into the evaluation run directory:

```bash
python -m src.evaluation.evaluate_pusht \
  --agent-type ppo \
  --checkpoint runs/ppo/pusht_real_ppo_best.pt \
  --protocol canonical_v2 \
  --visualize-starts
```

The evaluator resets the environment with the exact selected episode seeds before policy rollout.
The PNG places a colored X at every initial block centroid and draws a short ray in its initial
orientation. Colors identify the six strata; rings around the goal centroid show the 70- and
140-pixel translation thresholds and the 200-pixel sampling cap. The background is the average of
all reset renders: static workspace and goal pixels remain visible while the individual agent and
block bodies fade. This extra reset pass does not take actions or alter the deterministic evaluation
episodes.

## Encoder resolution

Latent BC/PPO evaluation uses
`$STABLEWM_HOME/checkpoints/pusht/lewm_object.ckpt` by default. An explicit local artifact can be
selected with `--encoder-checkpoint PATH`. Filesystem paths saved in PPO training metadata are kept
only as provenance and are not dereferenced during evaluation because they may refer to another
machine.
