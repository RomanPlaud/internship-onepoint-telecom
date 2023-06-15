from sknetwork.data import load_netset
import numpy as np
from sklearn.model_selection import train_test_split


def get_labels(labels_hierarchy, names_labels_hierarchy, depth=1, sep='|||'):
    names_depth = [sep.join(name.split(sep)[:depth]) for name in names_labels_hierarchy]
    names_depth_index = {name: i for i, name in enumerate(np.unique(names_depth))}
    index = np.array([names_depth_index[name] for name in names_depth])
    labels = index[labels_hierarchy]
    names_labels = np.array(list(names_depth_index))
    return np.array(labels), names_labels

def get_data_preprocessed(dataset_name='wikivitals', method='bow', depth=2, split=[0.7, 0.1, 0.2]):

    dataset = load_netset(dataset_name)
    
    if method == 'bow':
        labels_hierarchy = dataset.labels_hierarchy
        names_labels_hierarchy = dataset.names_labels_hierarchy

        labels, names_labels = get_labels(labels_hierarchy, names_labels_hierarchy, depth=depth)
        X = dataset.biadjacency

        ## add a seed to keep the same results
        np.random.seed(42)
        ## split the dataset into train validation and test using sklearn
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=split[2], random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split[1], random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test, len(set(labels))
    

