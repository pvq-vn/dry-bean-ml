import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X_arr = np.array(X, dtype=float)
        self.mean_ = np.mean(X_arr, axis=0)
        self.scale_ = np.std(X_arr, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None: raise ValueError()
        X_arr = np.array(X, dtype=float)
        X_scaled = (X_arr - self.mean_) / self.scale_
        
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)