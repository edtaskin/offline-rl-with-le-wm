from src.representations.history import LatentHistory, temporal_ensemble_action


def history_indices(index, episode_start, frame_stack, frame_stride):
    return [
        max(episode_start, index - offset * frame_stride)
        for offset in range(frame_stack - 1, -1, -1)
    ]


def action_chunk_indices(index, episode_end, action_chunk_size):
    return [min(episode_end - 1, index + offset) for offset in range(action_chunk_size)]


class FeatureHistory(LatentHistory):
    """BC compatibility wrapper returning a batched latent stack."""

    def stacked(self, device=None):
        return super().stacked(device=device).unsqueeze(0)
