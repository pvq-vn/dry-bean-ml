import numpy as np

class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)

        return self

    def transform(self, y):
        if self.classes_ is None: raise ValueError()

        y_array = np.asarray(y)
        class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}

        try:
            encoded = np.array(
                [class_to_index[label] for label in y_array],
                dtype=int
            )
        except KeyError as exc:
            raise ValueError(
                f"Unknown label encountered: {exc.args[0]}"
            ) from exc

        return encoded

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        if self.classes_ is None:  raise ValueError()

        y_array = np.asarray(y)

        if not np.issubdtype(y_array.dtype, np.integer):
            raise ValueError()

        if np.any(y_array < 0) or np.any(y_array >= len(self.classes_)):
            raise ValueError()

        return self.classes_[y_array]