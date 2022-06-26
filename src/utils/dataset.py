import os 
import pandas as pd
from torch.utils.data import Dataset

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

# test dataset performance
if __name__ == '__main__':
    DATAPATH = os.path.join('data', 'olid-training-v1.0.tsv')

    dataset = OLID(DATAPATH)

    print(dataset[10])
    print(len(dataset))
    print(dataset.data.head())