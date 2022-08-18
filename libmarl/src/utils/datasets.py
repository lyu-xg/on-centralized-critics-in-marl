import numpy as np
from collections.abc import Iterable
from torch.utils.data import Dataset, DataLoader


class NonSequentialSingleTDDataset(Dataset):
    def __init__(self, s, a, r, t) -> None:
        super().__init__()
        self.s, self.a, self.r, self.t = s, a, r, t
        assert len(s) == len(a) == len(r) == len(t)

    def __getitem__(self, index: int):
        return (
            self.s[index],
            self.a[index],
            self.r[index],
            self.t[index],
            self.s[index + 1],
        )

    def __len__(self) -> int:
        return len(self.s) - 1


class NonSequentialSingleDataset(Dataset):
    def __init__(self, *arrays) -> None:
        super().__init__()
        # just in case if we have an episode with one single step
        self.arrays = [a if isinstance(a, Iterable) else [a] for a in arrays]
        assert all(len(a) == len(self) for a in arrays)

    def __getitem__(self, index: int):
        return [a[index] for a in self.arrays]

    def __len__(self):
        return len(self.arrays[0])


class NonSequentialParallelDataset(Dataset):
    """
     * ``N`` - number of parallel environments
     * ``T`` - number of time steps explored in environments

    Dataset that flattens ``N*T*...`` arrays into ``B*...`` (where ``B`` is equal to ``N*T``) and returns such rows
    one by one. So basically we loose information about sequence order and we return
    for example one state, action and reward per row.

    It can be used for ``Model``'s that does not need to keep the order of events like MLP models.

    For ``LSTM`` use another implementation that will slice the dataset differently
    """

    def __init__(self, *arrays: np.ndarray) -> None:
        """
        :param arrays: arrays to be flattened from ``N*T*...`` to ``B*...`` and returned in each call to get item
        """
        super().__init__()
        self.arrays = [array.reshape(-1, *array.shape[2:]) for array in arrays]

    def __getitem__(self, index):
        return [array[index] for array in self.arrays]

    def __len__(self):
        return len(self.arrays[0])
