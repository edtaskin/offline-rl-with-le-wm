# Offline RL with LeWM
### Can a policy be improved inside a JEPA world model, without touching the environment?

Deep Learning Lab project, University of Freiburg, SoSe 2026

**Summary:** [LeWorldModel](https://github.com/lucas-maes/le-wm) (LeWM) is a joint-embedding
predictive architecture that learns PushT dynamics in a 192-d latent space from pixels. If those
dynamics are good enough, then everything an RL agent needs — states, transitions, rewards,
termination — can be produced *inside* the model, and policy improvement becomes an offline
procedure: no simulator, no environment steps, only a frozen world model and expert data. This
repository builds that pipeline on PushT and then tries to falsify it. A latent behavioural-cloning
prior is trained on the frozen LeWM encoder, and chunk-level PPO is run twice from the same prior
under one evaluation protocol: once in the real simulator, once entirely in imagination, where a
frozen probe supplies reward and success and a learned bridge maps LeWM's predicted latents back
into the policy's input space. Dream PPO reaches **72.7% ± 3.5** success on the real task having
consumed **zero** environment transitions, against **77.8% ± 4.8** for the real-env agent it is
matched against and **50.7%** for the BC prior both started from. The gap is small; the reason it
exists is not. The imagined reward is systematically optimistic — 84.5% imagined success against
57.1% real for the same weights — and 53% of the successes PPO was rewarded for never happened when
the same action chunks were replayed in the simulator.

<p align="center">
   <b>[ <a href="https://docs.google.com/presentation/d/1-oNcFspcV0CBroVyhtFS_ti9GM_A7fOuP8SgzajbwqY/edit?usp=sharing">Poster</a> | <a href="https://huggingface.co/offline-rl-with-le-wm">Checkpoints</a> | <a href="https://github.com/lucas-maes/le-wm">LeWM</a> | <a href="https://le-wm.github.io/">LeWM website</a> ]</b>
</p>

## Pipeline

Everything downstream of the encoder is frozen-encoder work: LeWM's ViT is never fine-tuned, and the
policy consumes raw CLS tokens (192-d) stacked over a dilated history.

```
                        expert PushT demonstrations (224 px)
                                        |
              +-------------------------+-------------------------+
              |                                                   |
      frozen LeWM ViT                                     frozen LeWM predictor
              |                                                   |
      raw CLS latent (192-d)                        projected latent (dynamics space)
              |                                                   |
         BC prior (MLP)                       decoder / de-projector    probes
              |                                (bridge back to CLS)   (reward, success)
              |                                                   |
              +--> PPO in the real env  (1M env steps)             |
              |                                                    |
              +--> PPO in imagination ("dream")  <-----------------+
                                        (0 env steps)
```

* **Encoder** ([`src/representations/lewm.py`](src/representations/lewm.py)) — the frozen LeWM ViT
  shared by BC, PPO and evaluation, so every agent sees exactly the same observation space.
* **BC prior** ([`src/bc/train_bc_latent.py`](src/bc/train_bc_latent.py)) — action-chunked latent
  behavioural cloning; it initialises every PPO run so the arms differ only in where experience
  comes from.
* **Real-env PPO** ([`src/ppo/train.py`](src/ppo/train.py)) — chunk-level PPO on `swm/PushT-v1`,
  the interaction-paying reference.
* **Dream PPO** ([`src/ppo/train_lewm.py`](src/ppo/train_lewm.py)) — the same PPO, rolled out
  through LeWM's predictor. One PPO transition = one predictor step = 5 env steps. `env.step` is
  never called.
* **Bridge** ([`src/representations/deprojector.py`](src/representations/deprojector.py)) — LeWM
  predicts in *projected* latent space; the policy reads raw CLS. Either decode the imagined latent
  to a frame and re-encode it, or learn the inverse map directly.
* **Probes** ([`scripts/probes/`](scripts/probes/)) — frozen classifiers on the imagined latent
  supply reward and episode termination. They are the only thing standing in for the simulator's
  reward function, and RQ2 is about how far that can be trusted.

## Results

The canonical protocol for new runs is one 150-episode suite sampled from master seed 42, fixed
target, block-start radius 200, 224 px observations, 300 max env steps, and open-loop chunk
execution. The results below predate that seed-sampling correction and should be regenerated before
being compared with new runs. PPO rows are `final.pt` (no checkpoint selection at all), mean ± std
over training seeds 1–3, all started from the same BC prior.

<div align="center">

| Agent | reward | env steps for training | success |
|:---|:---:|:---:|:---:|
| BC prior (`pusht-bc-raw-cls`) | — | 0 | 50.7 |
| PPO — real env | sparse | 1M | **77.8 ± 4.8** |
| PPO — real env | dense (block-pose shaping) | 1M | **80.7 ± 2.3** |
| PPO — in LeWM | sparse (probe-declared success) | **0** | 72.7 ± 3.5 |
| PPO — in LeWM | dense (time-to-success classifier) | **0** | 66.9 ± 4.8 |

</div>

The reward axis is **not symmetric**: "dense" means block-pose shaping in the real env and a learned
classifier in the dream. The matched dense pair needs a `block_rel_objective` regression probe that
is not published yet — read the dense column as two shaping schemes, not one comparison.

"Zero env steps" is a property of the *training loop*, and it is counted rather than asserted: every
checkpoint records `env_steps_consumed`, and `--selection dream` ranks checkpoints on imagined
success over held-out expert episodes so that even model selection is simulator-free. One caveat
this campaign does not escape: the shared BC prior was itself picked by real-env success, so a dream
run started from it is not interaction-free *end to end*. Use `pusht_bc_raw_cls_final.pth` for the
strict claim.

A separate control ([`scripts/campaign/run_legacy_prior_control.sh`](scripts/campaign/run_legacy_prior_control.sh))
re-runs the real-env arms from the earlier 256-wide prior that produced this project's best number
(92.0%), to separate "better prior" from "lucky seed". It is still running; seeds 2 and 3 score 85.3
and 92.7, so the prior looks like a real effect, but three seeds are not yet in.

## Does the world model tell the truth?

Success in the dream is declared by a frozen probe reading a latent no simulator ever corrects —
the textbook setup for model exploitation. Three measurements, one per failure mode:

<div align="center">

| Question | Measurement | Result |
|:---|:---|:---:|
| Is imagined success optimistic? | imagined vs real held-out success, same weights | 84.5% vs 57.1% (**+27.5 pp**), Spearman ρ = 0.39 |
| Does selecting on it cost anything? | real success given up by ranking on the imagined metric | **12.5 pp** regret |
| Were the rewarded successes real? | probe-declared successes replayed in the simulator | 41 / 87 confirmed → **52.9% hallucinated** |

</div>

The optimism rows pool 56 paired measurements over 2 seeds; the hallucination row is seed 1, 96 dream
episodes, replayed open-loop from the same dataset start states.

The probe is not uniformly wrong — it is wrong *with depth*. Its false-positive rate is 0.0% out to
20 env steps of imagination, 1.2% at 40, and 16.6% at 100, against 2.0% when the same classifier
reads encoded ground-truth frames. Most of that is world-model drift, not probe error: LeWM's own
latent drifts 0.08 relative L2 after 5 env steps and 0.79 after 100. Dream episodes reach declared
success in ~7.5 predictor steps (≈38 env steps) on average, which is precisely where the curve
starts to bend.

Open-loop, the bridge is not the bottleneck: decoder and de-projector track the ground-truth CLS
latent almost identically (relative L2 0.100 vs 0.081 at 5 env steps, 0.383 vs 0.380 by 50), and the
de-projector gets there without decoding a frame or re-encoding it with the ViT, at roughly three
orders of magnitude less compute per imagined step. Trained through, the picture is less settled: on
the legacy prior the two bridges land within noise (74.7 vs 72.7 at seed 1), but on the campaign
prior the decoder scores 55.3 against the de-projector's 70.7 mean. That is either a prior × bridge
interaction or one unlucky run — the decoder side is n=1 per prior, and
[`run_decoder_bridge_control.sh`](scripts/campaign/run_decoder_bridge_control.sh) exists to fill in
seeds 2 and 3 and settle it.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # requirements-macos.txt on Apple silicon
cp .env.example .env                     # LE_WM_PATH, WANDB_API_KEY, HF_TOKEN
```

Clone [LeWM](https://github.com/lucas-maes/le-wm) next to this repo (or point `LE_WM_PATH` at it),
then fetch and convert the pretrained PushT encoder — the HF mirror ships a state dict, the loaders
want a serialized object checkpoint:

```bash
python -m scripts.download_lewm_checkpoint     # idempotent
python scripts/download_pusht_data.py          # expert demonstrations
```

The real-env arms need nothing further. The dream arms additionally want LeWM's HDF5 expert dataset
at `le-wm/models/datasets/pusht_expert_train.h5` (~46 GB, for episode starts) and the `objective_met`
classifier under `models/probes/pusht_lewm/`. The campaign scripts preflight all of it — encoder,
dataset, probes and every Hub destination — before the first run starts.

Trained policy artifacts are published through Hugging Face and kept out of Git.
`hf://<owner>/<repo>/<filename>` references check the requested Hub revision on each process start
and reuse the Hugging Face cache when its commit has not changed. Legacy paths under `checkpoints/`
resolve to registered Hub artifacts and are never read from the local filesystem. Note that `hf://`
URIs carry no revision component and always resolve to `main`, so re-publishing over an existing
path orphans every reference to the old artifact.

## Data

Generate the LeWM-resolution expert dataset once after downloading the original PushT
demonstrations. This restores each recorded simulator state and renders it directly at 224×224; it
does not interpolate the stored 96×96 pixels.

```bash
python scripts/regenerate_pusht_expert.py \
  --dataset data/expert_trajectories/pusht_expert.npz \
  --output-dataset data/expert_trajectories/pusht_expert_224.npz \
  --verify-n 100
```

The generator uses a disk-backed image buffer. During the final save, allow space for both that
temporary buffer and the output dataset. Add `--compressed` if disk space matters more than
generation and load time.

## Behavioural cloning

Train the latent BC policy with the configuration used for the current checkpoint:

```bash
python -m src.bc.train_bc_latent \
  --data_path data/expert_trajectories/pusht_expert_224.npz \
  --checkpoint_path runs/bc/pusht_latent_bc.pth \
  --observation-resolution 224 \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.001 \
  --hidden_dim 256 \
  --frame_stack 3 \
  --frame_stride 5 \
  --action_chunk_size 5 \
  --seed 42 \
  --num_workers 0 \
  --deterministic \
  --log_interval 10 \
  --save_interval 100 \
  --push_to_hf \
  --hf_repo_id offline-rl-with-le-wm/bc/pusht_latent_bc
```

Evaluate on the fixed-target task. Passing `--block-start-radius 200` matches the PPO training start
distribution; omit it for unrestricted block starts around the same fixed target.

```bash
python -m src.evaluation.evaluate_pusht \
  --agent-type bc \
  --checkpoint hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls/pusht_bc_raw_cls_best.pth \
  --stats hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls/pusht_bc_raw_cls_best_stats.pth \
  --training-observation-resolution 224 \
  --block-start-radius 200 \
  --episodes 150 \
  --max-episode-steps 300 \
  --video \
  --seed 42
```

### The evaluation protocol

Every agent in this repo is scored by one environment-owned evaluator, and the numbers above are
only comparable because of it. Each evaluation creates a timestamped directory under
`runs/evaluations/`. By default it runs 150 episodes in one evaluation. `--seed` is a single master
seed from which the evaluator samples a reproducible set of unique episode seeds across the 32-bit
range; sampled values are at least seven apart to avoid the underlying PushT reset's adjacent-seed
collisions. The complete episode-seed list and summary land in `metrics.json`; with `--video`,
episode videos are saved under `videos/`. Use `--output-root` to change the parent directory and
`--run-name` to append a readable label.

Policy observations render at 224×224 by default, stored in `metrics.json` as
`config.observation_resolution`; pass `--observation-resolution 96` to reproduce the earlier
low-resolution protocol. `--video-resolution` controls only saved-video scaling and does not affect
policy inputs. BC stats and PPO checkpoints record their native training observation resolution, and
evaluation rejects a mismatch by default. For older artifacts that predate this metadata, provide
`--training-observation-resolution`; use `--allow-resolution-mismatch` only for an intentional
resolution-transfer experiment.

## PPO in the real environment

```bash
python src/ppo/train.py \
  --bc_checkpoint hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls/pusht_bc_raw_cls_best.pth \
  --bc_stats hf://offline-rl-with-le-wm/bc/pusht-bc-raw-cls/pusht_bc_raw_cls_best_stats.pth \
  --hidden_dim 256 \
  --frame_stack 3 \
  --frame_stride 5 \
  --action_chunk_size 5 \
  --observation-resolution 224 \
  --fixed_target \
  --log_interval 10 \
  --save_interval 10 \
  --eval_interval 10 \
  --total_timesteps 1000000 --num_envs 8 --num_chunks 64 \
  --push_to_hf \
  --hf_repo_id offline-rl-with-le-wm/ppo
```

PPO uses the same environment-owned, fixed-target evaluator as BC; the checkpoint and agent type
select the PPO adapter.

```bash
python -m src.evaluation.evaluate_pusht \
  --agent-type ppo \
  --checkpoint hf://offline-rl-with-le-wm/ppo/best.pt \
  --block-start-radius 200 \
  --episodes 150 \
  --max-episode-steps 300 \
  --seed 42 \
  --video
```

## RQ1: can policy improvement happen inside the world model?

[`src/ppo/train_lewm.py`](src/ppo/train_lewm.py) runs the same chunk-level latent PPO as
[`src/ppo/ppo.py`](src/ppo/ppo.py), but rolls out inside LeWM instead of the simulator: expert data
supplies episode starts, the frozen world model supplies dynamics, the bridge maps back to the
policy's latent space, and a frozen probe supplies reward and success. No `env.step` is called.

Two things make "zero environment interaction" a checkable claim rather than a description:

* **Interaction is counted.** Every checkpoint records `env_steps_consumed`
  (`train_env_steps + eval_env_steps`). Held-out evaluation counts, because picking `best.pt` by
  real-env success is interaction even though it trains nothing. Dream runs report training steps as
  0 by construction.
* **Selection can be interaction-free.** `--selection dream` ranks checkpoints by imagined success on
  expert episodes held out at *episode* granularity (`--dream_val_fraction`), so `best.pt` is chosen
  without a simulator. Use `--selection real` for the interaction-paying contrast, and `final.pt` as
  the no-selection control.

Train one dream agent whose entire interaction budget is zero:

```bash
python -m src.ppo.train_lewm \
  --exp_name rq1_dream_ppo --seed 1 \
  --fixed_target --reward_mode sparse \
  --block_start_near_goal --block_start_radius 200 \
  --total_timesteps 1000000 --num_envs 8 --num_chunks 64 \
  --eval_interval 0 \
  --selection dream --dream_eval_interval 25 --dream_eval_episodes 96 \
  --snapshot_interval 25
```

Add `--bridge deprojector` to imagine in latent space instead of through pixels (the default is
`decoder`).

`--snapshot_interval` keeps permanent `snapshot_step<env_steps>_it<iteration>.pt` checkpoints;
unlike `latest.pt` they are never overwritten, which is what makes an interaction-budget curve
possible after the fact. It applies to [`src/ppo/train.py`](src/ppo/train.py) too.

### Running the campaign

One BC prior, one protocol, every PPO arm — real and dream arms land in the same table without a
"different prior" footnote:

```bash
bash scripts/campaign/run_rawcls_grid.sh                    # 4 arms x seeds 1 2 3
bash scripts/campaign/run_rawcls_grid.sh --only dream_sparse
bash scripts/campaign/run_rawcls_grid.sh --dry-run          # print commands only
```

Arms are `real_sparse`, `real_dense`, `dream_sparse`, `dream_dense`. Runs are sequential (every arm
wants the GPU and the frozen ViT), each pushes `best.pt`, `final.pt` and — for dream arms —
`selection_log.jsonl` to the Hub, and every destination is checked for collisions before training
starts so nothing existing is overwritten. `--exp-prefix` renames run dirs, W&B runs and the Hub
folder so a second prior can reuse the script;
[`run_legacy_prior_control.sh`](scripts/campaign/run_legacy_prior_control.sh) is that case.

The older three-seed RQ1 harness still works and additionally draws the interaction-budget curve:

```bash
bash scripts/rq1/run_campaign.sh --seeds "1 2 3"   # trains real-env PPO and dream PPO
python -m scripts.rq1.evaluate_grid --seeds 1 2 3  # canonical 150-episode evaluation
python -m scripts.rq1.budget_curve  --seeds 1 2 3  # real-PPO snapshots vs env steps
python -m scripts.rq1.report                       # table.md + figures + summary.json
```

Training is not resumable, but evaluation is: both evaluation scripts append to `runs/rq1/*.jsonl`
and skip checkpoints already scored under the same protocol, so an interrupted campaign continues
where it stopped.

`evaluate_grid.py` scores three checkpoints per PPO run, which is what makes the selection story
explicit:

| variant | what selected it | interaction it required |
|---|---|---|
| `final` | nothing — the last checkpoint | training only (0 for dream PPO) |
| `best` | the run's own `--selection` rule | 0 for `--selection dream` |
| `best_real_sel` | real held-out success, reconstructed post-hoc from `selection_log.jsonl` | training + every eval step up to that checkpoint |

`report.py` reports **required** env steps, not merely consumed ones: a dream run launched with
`--record-real-eval` also spends simulator steps, but those are RQ2 instrumentation that never feeds
selection, so they are listed separately as diagnostic rather than charged to the claim.

## RQ2: is the imagined reward trustworthy?

The dream reward *and* the dream episode's termination both come from one frozen probe reading a
latent the simulator never corrects. That is the textbook setup for model exploitation, and the run
is already instrumented to measure it — three scripts, one per piece of evidence:

```bash
# 1. optimism: imagined vs real held-out success, same weights, same x-axis
python -m scripts.rq2.optimism_gap --seeds 1 2 3

# 2. trust vs depth: probe false-positive rate over a 5 -> 100 env-step horizon
python -m scripts.rq2.probe_horizon --seeds 1 2 3

# 3. the frames PPO believed were successes, checked against the simulator
python -m scripts.rq2.dream_success_gallery --seed 1 --episodes 96 --verify-in-sim
```

* **`optimism_gap.py`** needs `--record-real-eval` on the dream run: the trainer then writes both
  metrics into `selection_log.jsonl` at the same cadence for the same weights. Reports the level gap,
  the rank agreement (Spearman ρ), and the real success given up by selecting on the imagined metric.
* **`probe_horizon.py`** resolves the `objective_met` classifier exactly the way `LeWMDreamWorld`
  does and pins [`scripts/probes/probe_rollouts_pusht.py`](scripts/probes/probe_rollouts_pusht.py) to
  that file, so the curve describes the probe PPO actually optimized against. It plots the imagined
  false-positive rate against the *encoded ground-truth* rate, which separates probe error from
  world-model drift, and marks the horizon the agent actually lives in.
* **`dream_success_gallery.py`** rolls the trained policy in the dream, catches probe-declared
  successes, and decodes the latent the probe made that call on. `--verify-in-sim` replays the same
  action chunks in the simulator from the same dataset start state, turning "these look wrong" into a
  hallucination rate. The replay is open-loop by design — the question is whether the *imagined
  trajectory* was real, not how good the policy is (that is RQ1's job) — and its
  `mean_anchor_pixel_error` reports how exactly the simulator reproduced the dream's start frame, so
  an unreliable verdict is visible rather than silent.

## Bridging LeWM's latent space back to the policy

LeWM's predictor emits latents in the *projected* space (`projector(cls)`), while the BC and PPO
policies consume the raw ViT CLS token. Imagination has to close that gap. The original route goes
through pixels: decode the imagined latent to a frame, re-encode the frame with the ViT. The
de-projector does the same job directly in latent space — no image reconstruction, no ViT pass over
synthetic frames — and like the decoder and the probes it is trained offline from expert data at
zero environment cost.

```bash
python -m scripts.deprojector.train_deprojector_pusht      # train the latent-space bridge
python -m scripts.deprojector.verify_parity                # decoder vs de-projector, matched seeds
python -m scripts.deprojector.bridge_horizon               # bridge error vs imagination depth
python -m scripts.deprojector.probe_trust_horizon          # probe FPR vs imagination depth
```

## Checkpoints

All artifacts live under [`hf.co/offline-rl-with-le-wm`](https://huggingface.co/offline-rl-with-le-wm)
and are referenced from the CLI as `hf://offline-rl-with-le-wm/<repo>/<file>`.

<div align="center">

| Artifact | Path | Notes |
|:---|:---|:---|
| BC prior (campaign) | `bc/pusht-bc-raw-cls/` | 512-wide raw CLS, 1000 epochs; the prior for every arm in the results table |
| BC prior (legacy) | `bc/pusht-bc-raw-cls-256-legacy/` | 256-wide, 100 epochs; the prior behind the 92% real-env PPO. Its stats file predates `observation_resolution` — evaluate with `--training-observation-resolution 224` |
| PPO agents | `ppo/` | `rawcls_bc_best/<arm>/seed<N>/` for the campaign, `rawcls256_legacy/` for the prior control |
| Latent decoder | `decoder_lewm_pusht/` | pixel bridge for imagination |
| De-projector | `deprojector_lewm_pusht/` | latent-space bridge |
| Success probe | `probes/is_objective_met_probe_baseline.pt` | declares success and termination in the dream |
| Reward classifier | `dense_reward_classifier/` | time-to-success heads used by `--reward_mode dense` |

</div>

## Repository layout

```
src/representations/   frozen LeWM encoder, dilated history, de-projector
src/bc/                latent behavioural cloning
src/ppo/               chunk-level latent PPO — train.py (real env), train_lewm.py (imagination)
src/evaluation/        the one evaluation protocol every agent is scored under
src/envs/              PushT wrappers (fixed target, block-start distribution)
scripts/probes/        reward / success classifiers on LeWM latents
scripts/decoder/       latent -> pixel decoder and rollout diagnostics
scripts/deprojector/   latent-space bridge, parity and horizon diagnostics
scripts/campaign/      one prior, one protocol, every PPO arm
scripts/rq1/           interaction-budget campaign, evaluation grid, report
scripts/rq2/           optimism gap, probe horizon, hallucination gallery
scripts/ablations/     LeWM visual-robustness ablation
```

## Acknowledgements

Built on [LeWorldModel](https://github.com/lucas-maes/le-wm) by Lucas Maes, Quentin Le Lidec, Damien
Scieur, Yann LeCun and Randall Balestriero, and on
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management
and evaluation. The PushT expert demonstrations come from
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/).
