"""Chunk-level latent PPO fine-tuning on ``swm/PushT-v1``.

A frozen LeWM encoder turns frames into latents; a BC-initialized chunk policy
is fine-tuned with PPO. See ``config.py`` for hyperparameters and the entry
point ``python -m src.ppo.train``. Evaluate BC and PPO checkpoints with
``python -m src.evaluation.evaluate_pusht``.
"""
