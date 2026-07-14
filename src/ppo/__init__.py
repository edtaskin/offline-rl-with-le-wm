"""Chunk-level latent PPO fine-tuning on ``swm/PushT-v1``.

A frozen LeWM encoder turns frames into latents; a BC-initialized chunk policy
is fine-tuned with PPO. See ``config.py`` for hyperparameters and the entry
points ``python -m src.ppo.train`` / ``python -m src.ppo.evaluate``.
"""
