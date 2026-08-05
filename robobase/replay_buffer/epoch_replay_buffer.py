import traceback

import numpy as np
import torch
from torch.utils.data import Dataset

from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer


class EpochReplayBuffer(UniformReplayBuffer, Dataset):
    """Finite shuffled epoch iterator for offline/controller training.

    This mirrors mobile-genima's controller replay path: each epoch shuffles the
    available transition indices and yields full batches until exhausted. Batch
    construction intentionally delegates to UniformReplayBuffer.sample(), so ACT
    action chunks stay sliding windows, e.g. idx..idx+action_sequence-1, and the
    explicit padding mask from UniformReplayBuffer is preserved.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_start = 0
        self._epoch_indices = []

    @property
    def length(self):
        return np.minimum(self._add_count.value, self._replay_capacity)

    def _sample(self, global_index=None):
        """Compatibility helper; keep sliding/padding semantics if called.

        mobile-genima carried a separate _sample implementation, but its
        __next__ path calls sample(), not _sample(). Delegating here avoids a
        second action-chunk implementation drifting out of sync.
        """
        return self.sample_single(global_index)

    def __next__(self):
        if self.batch_start >= len(self._epoch_indices):
            raise StopIteration

        end = self.batch_start + self.batch_size
        if end > len(self._epoch_indices):
            raise StopIteration

        batch_indices = self._epoch_indices[self.batch_start:end]
        self.batch_start += self.batch_size
        return self.sample(batch_size=len(batch_indices), indices=batch_indices)

    def __iter__(self):
        try:
            self._try_fetch()
        except Exception as exc:
            print(exc)
            traceback.print_exc()

        self.batch_start = 0
        shuffled = torch.randperm(self.length)
        self._epoch_indices = [
            i.item()
            for i in shuffled
            if i.item() in self._global_idxs_to_episode_and_transition_idx
        ]
        return self
