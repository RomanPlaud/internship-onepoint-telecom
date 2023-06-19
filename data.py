from sknetwork.data import load_netset
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def get_labels_v2(labels_hierarchy, names_labels_hierarchy, hierarchy=True, depth=1, sep='|||'):
    names_depth = [sep.join(name.split(sep)[:depth]) for name in names_labels_hierarchy]
    names_depth_index = {name: i for i, name in enumerate(np.unique(names_depth))}
    index = np.array([names_depth_index[name] for name in names_depth])
    labels = index[labels_hierarchy]
    names_labels = np.array(list(names_depth_index))
    levels = [np.array(list(names_depth_index.values()))]

    if hierarchy:
        for i in range(depth-1, 0, -1):
            names_depth_i = [sep.join(name.split(sep)[:i]) for name in names_labels_hierarchy]
            names_depth_index_i = {name: i for i, name in enumerate(np.unique(names_depth_i))}

            names_depth_i = [sep.join(name.split(sep)[:i]) for name in names_labels]
            level = np.array([names_depth_index_i[name] for name in names_depth_i])
            levels.append(level)

    return np.array(labels), names_labels, np.array(levels)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, labels):
        self.X = X.toarray()
        self.labels = labels
    def __len__(self):
        return (self.X).shape[0]
    def __getitem__(self, index):
        # transform to tensor
        X = torch.from_numpy(self.X[index].astype('float32'))
        return X, self.labels[index]
    


def get_data_preprocessed(dataset_name='wikivitals', method='bow', depth=2, split=[0.85, 0.15]):

    dataset = load_netset(dataset_name)
    
    if method == 'bow':
        labels_hierarchy = dataset.labels_hierarchy
        names_labels_hierarchy = dataset.names_labels_hierarchy

        labels, names_labels, hierarchy = get_labels_v2(labels_hierarchy, names_labels_hierarchy, depth=depth)
        X = dataset.biadjacency

        dataset = Dataset(X, labels)

        train_set, test_set = torch.utils.data.random_split(dataset, split, generator=torch.Generator().manual_seed(42))


        return train_set, test_set, hierarchy
    
