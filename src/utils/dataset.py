import os 
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import random_split

class OLID(Dataset):
    def __init__(self, path):
        self.subtask_a = {
            'NOT': 0,   # not offensive
            'OFF': 1    # offensive
        }
        self.data = pd.read_csv(path, sep='\t').drop(['subtask_b', 'subtask_c'], axis=1)

    def __getitem__(self, index):
        data = self.data.iloc[index]

        return data['tweet'], self.subtask_a[data['subtask_a']]
    
    def __len__(self):
        return self.data.shape[0]

def get_dataset(path, split_size=[0.9, 0.05]):
    """
        Returns dataset splitted into train, validation and test sets.

        path: path to the dataset
        split_size: list of floats, representing the size of train, validation and test sets
    """

    assert len(split_size) == 2, 'split_size must be a list of length 2'
    assert all([isinstance(item, float) for item in split_size]), 'split_size must be a list of floats'
    assert sum(split_size) < 1, 'split_size must sum to less than 1'

    dataset = OLID(path)

    train_size = int(split_size[0] * len(dataset))
    valid_size = int(split_size[1] * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    return random_split(dataset, [train_size, valid_size, test_size])

# test dataset performance
if __name__ == '__main__':
    DATAPATH = os.path.join('data', 'olid-training-v1.0.tsv')

    dataset = OLID(DATAPATH)

    print(dataset[10])
    print(len(dataset))
    print(dataset.data.head())