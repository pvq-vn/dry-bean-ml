import numpy as np

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    classes = np.unique(np.concatenate([y_true, y_pred]))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    class_to_index = {
        cls: i for i, cls in enumerate(classes)
    }

    for true, pred in zip(y_true, y_pred):
        matrix[
            class_to_index[true],
            class_to_index[pred]
        ] += 1

    return matrix


def precision(y_true, y_pred, average="macro"):
    cm = confusion_matrix(y_true, y_pred)

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp

    result = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fp) != 0
    )

    if average == "macro":
        return np.mean(result)

    if average == "weighted":
        y_true = np.asarray(y_true).ravel()
        classes = np.unique(
            np.concatenate([
                np.asarray(y_true),
                np.asarray(y_pred).ravel()
            ])
        )
        weights = np.array([
            np.sum(y_true == cls)
            for cls in classes
        ])

        return np.average(result, weights=weights)

    if average is None:
        return result

    raise ValueError(
        "average must be 'macro', 'weighted', or None."
    )


def recall(y_true, y_pred, average="macro"):
    cm = confusion_matrix(y_true, y_pred)

    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp

    result = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(tp + fn) != 0
    )

    if average == "macro":
        return np.mean(result)

    if average == "weighted":
        y_true = np.asarray(y_true).ravel()
        classes = np.unique(
            np.concatenate([
                np.asarray(y_true),
                np.asarray(y_pred).ravel()
            ])
        )
        weights = np.array([
            np.sum(y_true == cls)
            for cls in classes
        ])

        return np.average(result, weights=weights)

    if average is None:
        return result

    raise ValueError(
        "average must be 'macro', 'weighted', or None."
    )


def f1_score(y_true, y_pred, average="macro"):
    p = precision(y_true, y_pred, average=None)
    r = recall(y_true, y_pred, average=None)

    result = np.divide(
        2 * p * r,
        p + r,
        out=np.zeros_like(p, dtype=float),
        where=(p + r) != 0
    )

    if average == "macro":
        return np.mean(result)

    if average == "weighted":
        y_true = np.asarray(y_true).ravel()
        classes = np.unique(
            np.concatenate([
                np.asarray(y_true),
                np.asarray(y_pred).ravel()
            ])
        )
        weights = np.array([
            np.sum(y_true == cls)
            for cls in classes
        ])

        return np.average(result, weights=weights)

    if average is None:
        return result

    raise ValueError(
        "average must be 'macro', 'weighted', or None."
    )