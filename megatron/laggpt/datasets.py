from gluonts.dataset.repository.datasets import get_dataset as get_gluonts_dataset
import random


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights.copy()
        self._rng = random.Random(seed)

    def __next__(self):
        
        data = None
        while not data:
            (index, ) = self._rng.choices(range(len(self._datasets)), weights=self._weights, k=1)
            try:
                data = next(self._datasets[index])
            except StopIteration:
                del self._datasets[index]
                del self._weights[index]
            if len(self._datasets) == 0:
                raise StopIteration
        
        return data
    

class CombinedDataset:
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)
    
    def __len__(self):
        return sum([len(ds) for ds in self._datasets])
    



def get_combined_dataset(dataset_names, rank, seed, test = False):

    if test:
        gluont_ds = [get_gluonts_dataset(i).test for i in dataset_names]
    else:
        gluont_ds = [get_gluonts_dataset(i).train for i in dataset_names]
    return CombinedDataset(gluont_ds, seed=rank + seed)

