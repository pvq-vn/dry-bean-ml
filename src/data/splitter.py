import numpy as np
import pandas as pd

def stratified_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    train_indices = []
    test_indices = []
    
    y_array = np.array(y)
    classes = np.unique(y_array)
    
    for cls in classes:
        cls_indices = np.where(y_array == cls)[0]
        np.random.shuffle(cls_indices)
        n_test = int(len(cls_indices) * test_size)
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])
        
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
        
    return X_train, X_test, y_train, y_test